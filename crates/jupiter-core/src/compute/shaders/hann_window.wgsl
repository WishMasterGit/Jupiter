// Hann window shader — applies a 2D Hann (raised-cosine) window to an image.
// out[row, col] = in[row, col] * 0.5*(1 - cos(TAU*row/h)) * 0.5*(1 - cos(TAU*col/w))

const TAU: f32 = 6.283185307179586;

struct Params {
    h: u32,
    w: u32,
}

@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;

    if row >= params.h || col >= params.w {
        return;
    }

    let idx = row * params.w + col;

    let wy = 0.5 * (1.0 - cos(TAU * f32(row) / f32(params.h)));
    let wx = 0.5 * (1.0 - cos(TAU * f32(col) / f32(params.w)));

    output[idx] = input[idx] * wy * wx;
}
