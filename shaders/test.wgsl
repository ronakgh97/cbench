@group(0) @binding(0)
var<storage, read> data_a: array<f32>;

@group(0) @binding(1)
var<storage, read> data_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&output)) { return; }
  output[i] = data_a[i] + data_b[i];
}