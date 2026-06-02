use aes_gcm::{AeadInOut, Aes256Gcm, KeyInit, Nonce, Tag};
use anyhow::Result;
use core::array::from_fn;
use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicBool, Ordering};
use rand::Rng;

/// A tiny fixed-size general purpose lock-free memory pool for type `T`.
/// Allocation atomically claims a free slot and returns a [`Block`].
/// Dropping the [`Block`] destroys the contained value and returns the
/// slot to the pool.
///
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

pub const NONCE_LEN: usize = 12;
pub const TAG_LEN: usize = 16;

#[inline(always)]
pub fn encrypt_buf(input: &[u8], output: &mut [u8], key: &[u8; 32]) -> Result<usize> {
    let plaintext_len = input.len();

    if output.len() < NONCE_LEN + plaintext_len + TAG_LEN {
        anyhow::bail!(
            "Output buffer too small, need at least {} bytes",
            NONCE_LEN + plaintext_len + TAG_LEN
        );
    }

    let (nonce_buf, data_tag) = output.split_at_mut(NONCE_LEN);
    let (data, tag_buf) = data_tag.split_at_mut(plaintext_len);

    // fill nonce fresh
    rand::rng().fill_bytes(nonce_buf);
    // copy plaintext
    data.copy_from_slice(input);

    let cipher = Aes256Gcm::new(key.into());
    let nonce = Nonce::try_from(&*nonce_buf)?;

    let auth_tag = cipher
        .encrypt_inout_detached(&nonce, b"", data.into())
        .map_err(|e| anyhow::anyhow!(e))?;

    tag_buf.copy_from_slice(&auth_tag);

    Ok(NONCE_LEN + plaintext_len + TAG_LEN)
}

#[inline(always)]
pub fn decrypt_buf(input: &[u8], output: &mut [u8], key: &[u8; 32]) -> Result<usize> {
    if input.len() < NONCE_LEN + TAG_LEN {
        anyhow::bail!("Ciphertext too short");
    }
    let ciphertext_len = input.len() - NONCE_LEN - TAG_LEN;

    if output.len() < ciphertext_len {
        anyhow::bail!(
            "Output buffer too small, need at least {} bytes",
            ciphertext_len
        );
    }
    let cipher = Aes256Gcm::new(key.into());

    // extract nonce & tag
    let nonce = Nonce::try_from(&input[..NONCE_LEN])?;
    let tag = Tag::try_from(&input[NONCE_LEN + ciphertext_len..])?;
    // ciphertext region only
    let data = &mut output[..ciphertext_len];
    data.copy_from_slice(&input[NONCE_LEN..NONCE_LEN + ciphertext_len]);

    cipher
        .decrypt_inout_detached(&nonce, b"", data.into(), &tag)
        .map_err(|e| anyhow::anyhow!(e))?;

    Ok(ciphertext_len)
}
