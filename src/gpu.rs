use anyhow::{Context, Result};
use wgpu::{Adapter, Instance};

pub async fn get_gpu_instance() -> Instance {
    Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: Default::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    })
}

pub async fn get_gpu_adapter(instance: Instance) -> Result<Adapter> {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .with_context(|| "Failed to request GPU adapter")?;
    Ok(adapter)
}

#[tokio::test]
async fn test_gpu() {
    let inst = get_gpu_instance().await;

    inst.enumerate_adapters(wgpu::Backends::all())
        .await
        .into_iter()
        .for_each(|adapter| {
            println!("Adapter: {}", adapter.get_info().name);
        });

    let adapter = get_gpu_adapter(inst).await.unwrap();
    println!("\nSelected Adapter: {}", adapter.get_info().name);
}
