use core::array::from_fn;
use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicBool, Ordering};

/// A tiny fixed-size general purpose lock-free memory pool for type `T`.
/// Allocation atomically claims a free slot and returns a [`Block`].
/// Dropping the [`Block`] destroys the contained value and returns the
/// slot to the pool.
///
/// Unsafety
/// For free\[i\] == true
/// - the slot is available for allocation
/// - the contents of `data[i]` are unspecified or uninitialized, and must not be read until allocated
///
/// For free\[i\] == false
/// - the slot is currently allocated and owned by some `Block`
/// - the contents of `data[i]` are initialized and valid, and can be read or written by the owner `Block` until it is dropped
pub struct StaticMemPool<T, const N: usize> {
    data: [UnsafeCell<MaybeUninit<T>>; N],
    free: [AtomicBool; N],
}

/// The pool contains no thread state, so `T: Send`
/// is required to ensures values stored in the pool may be transferred across threads safely.
unsafe impl<T: Send, const N: usize> Send for StaticMemPool<T, N> {}
unsafe impl<T: Send, const N: usize> Sync for StaticMemPool<T, N> {}

/// Each block is owned by a single thread, so `T: Send`
/// is required to ensure the value can be safely sent across threads if the block is moved.
unsafe impl<T: Send, const N: usize> Send for Block<'_, T, N> {}
unsafe impl<T: Sync, const N: usize> Sync for Block<'_, T, N> {}

#[allow(clippy::missing_safety_doc)]
/// A type whose all-zero bit pattern is a valid value.
/// Implementing this trait asserts that: `mem::zeroed::<T>()` would produce a valid instance of `T`.
/// This must remain true for all possible bitwise-zero values.
///
/// Examples:
/// - `u32`, `i64`, `f32` are valid
/// - `[T; N]` is valid when `T: Zeroable`
///
/// Non-examples:
/// - `String`
/// - `Vec<T>`
/// - `&T`
/// - `NonZeroU32`
pub unsafe trait Zeroable: Copy {}
unsafe impl Zeroable for u8 {}
unsafe impl Zeroable for u16 {}
unsafe impl Zeroable for u32 {}
unsafe impl Zeroable for u64 {}
unsafe impl Zeroable for usize {}
unsafe impl Zeroable for i8 {}
unsafe impl Zeroable for i16 {}
unsafe impl Zeroable for i32 {}
unsafe impl Zeroable for i64 {}
unsafe impl Zeroable for isize {}
unsafe impl Zeroable for f32 {}
unsafe impl Zeroable for f64 {}
unsafe impl<T: Zeroable, const N: usize> Zeroable for [T; N] {}

/// Exclusive ownership of a pool slot.
///
/// A `Block` behaves similarly to a `Box<T>` backed by pool storage.
/// At most one live `Block` may exist for a slot at any time.
/// When dropped:
/// - the contained `T` is dropped
/// - the slot is returned to the pool
///
/// Users should never manually free pool slots; ownership is managed
/// entirely through RAII.
pub struct Block<'a, T, const N: usize> {
    pool: &'a StaticMemPool<T, N>,
    index: usize,
}

impl<T, const N: usize> StaticMemPool<T, N> {
    /// Initializes a new memory pool with all blocks marked as free and uninitialized garbage data
    pub fn init() -> Self {
        Self {
            data: from_fn(|_| UnsafeCell::new(MaybeUninit::uninit())),
            free: from_fn(|_| AtomicBool::new(true)),
        }
    }

    #[inline(always)]
    /// Attempts to allocate a block from the pool, writing `value` inside uninitialized block, returns `Some(Block)` or `None` if the pool is exhausted
    pub fn try_alloc(&self, value: T) -> Option<Block<'_, T, N>> {
        for (index, flag) in self.free.iter().enumerate() {
            // if the block is free, attempt to mark it as occupied (false) atomically
            if flag.swap(false, Ordering::AcqRel) {
                unsafe {
                    // write the passed value into the uninitialized slot safely
                    (*self.data[index].get()).write(value);
                }
                return Some(Block { pool: self, index });
            }
        }
        None
    }

    #[inline(always)]
    /// Attempts to allocate a block and initialize it using `init`. (same semantics as `try_alloc`)
    /// The initializer is only called after a slot has been successfully claimed, returns `None` if the pool is exhausted.
    pub fn try_alloc_with<F>(&self, init: F) -> Option<Block<'_, T, N>>
    where
        F: FnOnce() -> T,
    {
        for (index, flag) in self.free.iter().enumerate() {
            if flag.swap(false, Ordering::AcqRel) {
                unsafe {
                    (*self.data[index].get()).write(init());
                }
                return Some(Block { pool: self, index });
            }
        }

        None
    }

    #[inline(always)]
    /// Claims a block and zeroes the memory directly in the pool, zero copies.
    /// This is only safe if `T` is a type where a purely zeroed type is valid, so calling this on a pool of `String` or `Vec` is instant UB.
    pub fn try_alloc_zeroed(&self) -> Option<Block<'_, T, N>>
    where
        T: Zeroable,
    {
        for (index, flag) in self.free.iter().enumerate() {
            if flag.swap(false, Ordering::AcqRel) {
                // get a raw pointer to the memory inside the pool
                let ptr = self.data[index].get().cast::<T>();

                // write zeroes directly into that memory (like memset)
                unsafe { core::ptr::write_bytes(ptr, 0u8, 1) };
                return Some(Block { pool: self, index });
            }
        }
        None
    }

    #[inline(always)]
    fn free_block(&self, index: usize) {
        self.free[index].store(true, Ordering::Release);
    }
}

impl<T, const N: usize> Block<'_, T, N> {
    /// Returns an immutable reference to the block's data
    pub fn get(&self) -> &T {
        unsafe { (*self.pool.data[self.index].get()).assume_init_ref() }
    }

    /// Returns a mutable reference to the block's data
    pub fn get_mut(&mut self) -> &mut T {
        unsafe { (*self.pool.data[self.index].get()).assume_init_mut() }
    }
}

impl<T, const N: usize> Deref for Block<'_, T, N> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<T, const N: usize> DerefMut for Block<'_, T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

impl<T, const N: usize> Drop for Block<'_, T, N> {
    // every live Block corresponds to a slot
    // whose occupancy flag is false and whose storage contains a fully initialized T.
    // The pool invariant guarantees that no Block can exist for an uninitialized slot.
    fn drop(&mut self) {
        unsafe {
            // drop the inner value before freeing the block back to the pool
            (*self.pool.data[self.index].get()).assume_init_drop();
        }
        self.pool.free_block(self.index);
    }
}

#[tokio::test]
async fn test_mem_pool() -> anyhow::Result<()> {
    use blas_rs::lvl1::dot_unsafe;
    use blas_rs::utils::gen_fill;
    use std::sync::Arc;

    let mem_pool = Arc::new(StaticMemPool::<[f32; 4096], 32>::init());

    let mut handles = vec![];
    for _ in 0..32 {
        let mem_pool = mem_pool.clone();
        handles.push(tokio::task::spawn_blocking(move || {
            if let Some(mut block_1) = mem_pool.try_alloc_zeroed() {
                if let Some(mut block_2) = mem_pool.try_alloc_zeroed() {
                    let x_buf: &mut [f32; 4096] = &mut block_1;
                    gen_fill(x_buf);
                    let y_buf: &mut [f32; 4096] = &mut block_2;
                    gen_fill(y_buf);

                    let result = unsafe { dot_unsafe(4096, x_buf, 1, y_buf, 1) };
                    println!("Got both blocks, dot: {}", result);
                    true
                } else {
                    println!("Failed to get block 2");
                    false
                }
            } else {
                println!("Failed to get block 1");
                false
            }
        }));
    }

    let mut s = 0;
    let mut e = 0;
    for handle in handles {
        if handle.await? {
            s += 1;
        } else {
            e += 1;
        }
    }
    println!("Failed alloc: {}, Successful alloc: {}", e, s);

    Ok(())
}

#[tokio::test]
async fn test_mem_pool_capacity() -> anyhow::Result<()> {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    let pool = Arc::new(StaticMemPool::<[f32; 4096], 32>::init());
    static LIVE: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);

    let mut handles = Vec::new();

    for _ in 0..512 {
        let pool = pool.clone();

        handles.push(tokio::task::spawn_blocking(move || {
            if let Some(_a) = pool.try_alloc_zeroed() {
                let live = LIVE.fetch_add(1, Ordering::SeqCst) + 1;
                PEAK.fetch_max(live, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(5));
                LIVE.fetch_sub(1, Ordering::SeqCst);
            }
        }));
    }

    for handle in handles {
        handle.await?;
    }

    let peak = PEAK.load(Ordering::SeqCst);

    println!("Peak live allocations = {}", peak);
    assert!(peak <= 32);

    Ok(())
}

#[tokio::test]
async fn test_mem_pool_max_two_block_owners() -> anyhow::Result<()> {
    use std::sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    };

    let pool = Arc::new(StaticMemPool::<[f32; 4096], 32>::init());
    let success = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(32));

    let mut handles = Vec::new();

    for _ in 0..32 {
        let pool = pool.clone();
        let success = success.clone();
        let barrier = barrier.clone();

        handles.push(tokio::task::spawn_blocking(move || {
            let block_1 = pool.try_alloc_zeroed();
            let block_2 = pool.try_alloc_zeroed();

            if block_1.is_some() && block_2.is_some() {
                success.fetch_add(1, Ordering::SeqCst);
                // hold both blocks until everyone has attempted allocation.
                barrier.wait();
                true
            } else {
                barrier.wait();
                false
            }
        }));
    }

    for handle in handles {
        handle.await?;
    }

    let successes = success.load(Ordering::SeqCst);
    println!("Successful two-block owners = {}", successes);
    assert_eq!(successes, 16);

    Ok(())
}

#[tokio::test]
async fn test_mem_pool_stress() -> anyhow::Result<()> {
    use std::sync::Arc;

    let pool = Arc::new(StaticMemPool::<[f32; 256], 32>::init());

    let mut handles = Vec::new();

    for _ in 0..64 {
        let pool = pool.clone();

        handles.push(tokio::task::spawn_blocking(move || {
            for _ in 0..128_000 {
                let mut block = loop {
                    if let Some(b) = pool.try_alloc_zeroed() {
                        break b;
                    }
                    std::hint::spin_loop();
                };
                block[0] += 1.0;
                block[1] += 2.0;
                block[2] += 3.0;
            }
        }));
    }

    for handle in handles {
        handle.await?;
    }

    Ok(())
}
