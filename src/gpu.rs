use anyhow::{Context, Result};
use wgpu::{Adapter, Instance};

#[inline]
pub async fn get_gpu_instance() -> Instance {
    Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: Default::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    })
}

#[inline]
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

/// This is my 'hello world' version of gpu programming
/// This just mul two vector parallelly
#[tokio::test]
async fn test_gpu() -> Result<()> {
    use crate::rand::gen_fill;
    use std::time::Duration;
    use wgpu::DeviceDescriptor;
    use wgpu::util::BufferInitDescriptor;
    use wgpu::util::DeviceExt;

    let inst = get_gpu_instance().await;
    let adapter = get_gpu_adapter(inst).await?;
    let shader_wgsl = wgpu::include_wgsl!("../shaders/test.wgsl");

    println!("Selected Adapter: {}", adapter.get_info().name);
    let (device, queue) = adapter.request_device(&DeviceDescriptor::default()).await?;

    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;

    let mut data_a = vec![0.0f32; 1024 * 1024 * 32];
    let mut data_b = vec![0.0f32; 1024 * 1024 * 32];

    // Fill buffer with random SHIT
    gen_fill(&mut data_a, &pool);
    gen_fill(&mut data_b, &pool);

    drop(pool); // Nobody likes you :0

    let a_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Input Buffer A"),
        contents: to_bytes(&data_a),
        usage: wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::STORAGE,
    });

    let b_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Input Buffer B"),
        contents: to_bytes(&data_b),
        usage: wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::STORAGE,
    });

    let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: (data_a.len() * size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (data_a.len() * size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            // A
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // B
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // OUTPUT
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // Create a bind group to pass the input buffer to the .wgsl
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buffer.as_entire_binding(),
            },
        ],
        label: Some("Bind Group"),
    });

    let shader = device.create_shader_module(shader_wgsl);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("add"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Command Encoder"),
    });

    // Binding vow
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Test Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        // Bind the input buffer to the compute shader
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let wg_size = 256;
        let max_wg_per_dim = device.limits().max_compute_workgroups_per_dimension;
        // Calculate the number of workgroups needed to process the entire input buffer
        let num_wg = (data_a.len() as u32).div_ceil(wg_size).min(max_wg_per_dim);

        // Dispatch the compute shader
        compute_pass.dispatch_workgroups(num_wg, 1, 1);
    }

    // Copy from DEVICE to HOST for reading
    encoder.copy_buffer_to_buffer(
        &out_buffer,
        0,
        &staging_buffer,
        0,
        (data_a.len() * size_of::<f32>()) as u64,
    );

    // Get the submission index before submitting the command buffer, so we can wait for it later
    let idx = queue.submit(Some(encoder.finish()));

    let slice = staging_buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});

    // Wait for the GPU to finish processing the command buffer before trying to read the output buffer
    device.poll(wgpu::PollType::Wait {
        submission_index: Some(idx),
        timeout: Some(Duration::from_secs(10)),
    })?;

    let data = slice.get_mapped_range();

    let result: Vec<f32> = data
        .chunks_exact(size_of::<f32>())
        .map(from_bytes::<f32>)
        .collect();

    println!("First 10: {:?}", &result[..10]);

    drop(data);
    staging_buffer.unmap();

    Ok(())
}

#[inline(always)]
/// Utility function to convert a slice of any type into a byte slice
pub fn to_bytes<T>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, size_of_val(data)) }
}

#[inline(always)]
/// Utility function to convert a byte slice back into a reference of any type
pub fn from_bytes<T: Copy>(data: &[u8]) -> T {
    unsafe { *(data.as_ptr() as *const T) }
}
