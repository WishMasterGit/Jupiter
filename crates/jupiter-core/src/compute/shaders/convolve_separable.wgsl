// Separable convolution shader with a-trous (dilated) support.
// Two entry points:
//   convolve_rows — convolves each row of the image
//   convolve_cols — convolves each column of the image
//
// The `step` parameter controls dilation (step=1 for standard convolution,
// step=2 for the second a-trous level, etc.).
// Mirror boundary handling is used at edges.

struct Params {
    h:          u32,
    w:          u32,
    kernel_len: u32,
    step:       u32,
}

@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read>       kernel: array<f32>;
@group(0) @binding(3) var<uniform>             params: Params;

// Mirror-reflect coordinate into [0, limit-1].
fn mirror(coord: i32, limit: i32) -> i32 {
    var c = coord;
    if c < 0 {
        c = -c;
    }
    if c >= limit {
        c = 2 * limit - 2 - c;
    }
    // Clamp as a safeguard for very large overshoot.
    return clamp(c, 0, limit - 1);
}

@compute @workgroup_size(256)
fn convolve_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Each thread processes one pixel (row, col).
    let idx = gid.x;
    if idx >= params.h * params.w {
        return;
    }

    let row = idx / params.w;
    let col = idx % params.w;

    let half_k = i32(params.kernel_len) / 2;
    var sum = 0.0;

    for (var i = 0; i < i32(params.kernel_len); i = i + 1) {
        let offset = (i - half_k) * i32(params.step);
        let sc = mirror(i32(col) + offset, i32(params.w));
        sum = sum + input[row * params.w + u32(sc)] * kernel[i];
    }

    output[idx] = sum;
}

@compute @workgroup_size(256)
fn convolve_cols(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.h * params.w {
        return;
    }

    let row = idx / params.w;
    let col = idx % params.w;

    let half_k = i32(params.kernel_len) / 2;
    var sum = 0.0;

    for (var i = 0; i < i32(params.kernel_len); i = i + 1) {
        let offset = (i - half_k) * i32(params.step);
        let sr = mirror(i32(row) + offset, i32(params.h));
        sum = sum + input[u32(sr) * params.w + col] * kernel[i];
    }

    output[idx] = sum;
}
