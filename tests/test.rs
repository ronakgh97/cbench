use cbench::load::StaticMemPool;

#[tokio::test]
async fn test_mem_pool_basic() -> anyhow::Result<()> {
    use blas_rs::lvl1::dot_unsafe;
    use blas_rs::utils::Noise;
    use std::sync::Arc;

    let mem_pool = Arc::new(StaticMemPool::<[f32; 4096], 32>::init());
    let noise = Noise::init();

    let mut handles = vec![];
    for _ in 0..32 {
        let mem_pool = mem_pool.clone();
        let mut noise = noise.clone();
        handles.push(tokio::task::spawn_blocking(move || {
            if let Some(mut block_1) = mem_pool.try_alloc_zeroed() {
                if let Some(mut block_2) = mem_pool.try_alloc_zeroed() {
                    let x_buf: &mut [f32; 4096] = &mut block_1;
                    noise.fill_f32(x_buf);
                    let y_buf: &mut [f32; 4096] = &mut block_2;
                    noise.fill_f32(y_buf);

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
    assert!(
        peak <= 32,
        "Peak live allocations {} exceeds pool capacity",
        peak
    );

    Ok(())
}

#[tokio::test]
async fn test_mem_pool_max_block_owners() -> anyhow::Result<()> {
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
    assert!(
        successes <= 16,
        "Number of two-block owners {} exceeds half the pool capacity",
        successes
    );

    Ok(())
}

#[tokio::test]
async fn test_mem_pool_stress() -> anyhow::Result<()> {
    use blas_rs::lvl1::scal_unsafe;
    use blas_rs::utils::Noise;
    use std::sync::Arc;

    let pool = Arc::new(StaticMemPool::<[f32; 2048], 32>::init());
    let noise = Noise::init();
    let mut handles = Vec::new();

    for _ in 0..64 {
        let pool = pool.clone();
        let mut noise = noise.clone();
        handles.push(tokio::task::spawn_blocking(move || {
            for _ in 0..128_000 {
                let mut block = loop {
                    if let Some(b) = pool.try_alloc_zeroed() {
                        break b;
                    }
                    std::hint::spin_loop();
                };
                noise.fill_f32(&mut *block);
                unsafe { scal_unsafe(2048, 0.5, &mut *block, 1) };
            }
        }));
    }
    for handle in handles {
        handle.await?;
    }

    Ok(())
}
