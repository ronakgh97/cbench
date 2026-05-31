use core::array::from_fn;
use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicBool, Ordering};

/// A tiny fixed-size general purpose lock-free memory pool for type `T`
pub struct StaticMemPool<T, const N: usize> {
    data: [UnsafeCell<MaybeUninit<T>>; N],
    free: [AtomicBool; N],
}

// The pool is safe to share across threads if T is safe to send.
unsafe impl<T: Send, const N: usize> Sync for StaticMemPool<T, N> {}
unsafe impl<T: Send, const N: usize> Send for StaticMemPool<T, N> {}

unsafe impl<T: Send, const N: usize> Send for Block<'_, T, N> {}
unsafe impl<T: Sync, const N: usize> Sync for Block<'_, T, N> {}

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

    /// Attempts to allocate a block from the pool, placing `value` inside it, returns `Some(Block)` or `None` if the pool is exhausted
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
    for _ in 0..16 {
        let mem_pool = mem_pool.clone();
        handles.push(tokio::spawn(async move {
            if let Some(mut block_1) = mem_pool.try_alloc([0.0; 4096]) {
                if let Some(mut block_2) = mem_pool.try_alloc([0.0; 4096]) {
                    let x_buf: &mut [f32; 4096] = &mut block_1;
                    gen_fill(x_buf);
                    let y_buf: &mut [f32; 4096] = &mut block_2;
                    gen_fill(y_buf);

                    let result = unsafe { dot_unsafe(4096, x_buf, 1, y_buf, 1) };
                    println!("Result: {}", result);
                } else {
                    println!("Failed to get block 2");
                }
            } else {
                println!("Failed to get block 1");
            }
        }));
    }

    for handle in handles {
        handle.await?;
    }

    Ok(())
}
