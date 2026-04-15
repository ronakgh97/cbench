@group(0) @binding(0) var<storage, read> data_a: array<f32>;

@group(0) @binding(1) var<storage, read> data_b: array<f32>;

@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Hardcoded for my machine
const PASS_SIZE: u32 = 256 * 65536;

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) id: vec3<u32>) {
    var i = id.x;
    if (i >= arrayLength(&output)) { return; }
    while(i < arrayLength(&output)) {output[i] = data_a[i] * data_b[i]; i += PASS_SIZE;}
}