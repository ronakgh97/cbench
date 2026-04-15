mod bencher;
mod gpu;
mod load;
mod rand;

pub mod prelude {
    pub use crate::bencher::*;
    pub use crate::gpu::*;
    pub use crate::load::*;
    pub use crate::rand::*;
    pub use crate::*;
}

#[test]
fn test_cpu_id() {
    let (l1, l2, l3) = get_cache_size();

    println!("L1 Cache: {} KB", l1);
    println!("L2 Cache: {} KB", l2);
    println!("L3 Cache: {} KB", l3);
}

/// Retrieves the CPU cache sizes (L1, L2, L3) in KB
pub fn get_cache_size() -> (usize, usize, usize) {
    let mut l1 = 0;
    let mut l2 = 0;
    let mut l3 = 0;

    let mut i = 0;

    loop {
        let res = std::arch::x86_64::__cpuid_count(4, i);

        let cache_type = res.eax & 0x1F;
        if cache_type == 0 {
            break;
        }

        let level = (res.eax >> 5) & 0x7;

        let ways = ((res.ebx >> 22) & 0x3FF) + 1;
        let partitions = ((res.ebx >> 12) & 0x3FF) + 1;
        let line_size = (res.ebx & 0xFFF) + 1;
        let sets = res.ecx + 1;

        let size_kb = (ways * partitions * line_size * sets) as usize / 1024;

        match (level, cache_type) {
            (1, 1) => l1 = size_kb,
            (2, 3) => l2 = size_kb,
            (3, 3) => l3 = size_kb,
            _ => {}
        }

        i += 1;
    }

    (l1, l2, l3)
}
