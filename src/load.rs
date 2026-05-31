use std::array::from_fn;
use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};

/// A tiny fixed-size memory pool of `u8` for general purpose
pub struct StaticMemPool<const BLOCK_SIZE: usize, const BLOCKS: usize> {
    data: [UnsafeCell<[u8; BLOCK_SIZE]>; BLOCKS],
    free: [AtomicBool; BLOCKS],
}
unsafe impl<const B: usize, const N: usize> Sync for StaticMemPool<B, N> {}
unsafe impl<const B: usize, const N: usize> Send for StaticMemPool<B, N> {}
unsafe impl<const B: usize, const N: usize> Send for Block<'_, B, N> {}

pub struct Block<'a, const B: usize, const N: usize> {
    pool: &'a StaticMemPool<B, N>,
    index: usize,
}

impl<const B: usize, const N: usize> StaticMemPool<B, N> {
    pub fn new() -> Self {
        Self {
            data: from_fn(|_| UnsafeCell::new([0u8; B])),
            free: from_fn(|_| AtomicBool::new(true)),
        }
    }

    pub fn alloc(&self) -> Option<Block<'_, B, N>> {
        for (index, flag) in self.free.iter().enumerate() {
            if flag.swap(false, Ordering::AcqRel) {
                return Some(Block { pool: self, index });
            }
        }
        None
    }

    fn free_block(&self, index: usize) {
        self.free[index].store(true, Ordering::Release);
    }
}

impl<const B: usize, const N: usize> Block<'_, B, N> {
    pub fn get(&self) -> &[u8; B] {
        unsafe { &*self.pool.data[self.index].get() }
    }

    pub fn get_mut(&mut self) -> &mut [u8; B] {
        unsafe { &mut *self.pool.data[self.index].get() }
    }
}

impl<const B: usize, const N: usize> Deref for Block<'_, B, N> {
    type Target = [u8; B];
    fn deref(&self) -> &Self::Target {
        self.get()
    }
}
impl<const B: usize, const N: usize> DerefMut for Block<'_, B, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}
impl<const B: usize, const N: usize> Drop for Block<'_, B, N> {
    fn drop(&mut self) {
        self.pool.free_block(self.index);
    }
}
