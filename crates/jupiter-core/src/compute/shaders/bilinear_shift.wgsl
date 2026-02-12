// Bilinear interpolation shift shader.
// Shifts image by (dx, dy) using bilinear sampling.
// Source coordinate: src_y = row - dy, src_x = col - dx.
// Returns 0.0 for out-of-bounds pixels.

struct Params {
    h:  u32,
    w:  u32,
    dx: f32,
    dy: f32,
}

@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;

fn sample(row: i32, col: i32) -> f32 {
    if row < 0 || row >= i32(params.h) || col < 0 || col >= i32(params.w) {
        return 0.0;
    }
    return input[u32(row) * params.w + u32(col)];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;

    if row >= params.h || col >= params.w {
        return;
    }

    let src_y = f32(row) - params.dy;
    let src_x = f32(col) - params.dx;

    // Check if fully out of bounds (with one-pixel margin for interpolation).
    if src_y < -1.0 || src_y >= f32(params.h) || src_x < -1.0 || src_x >= f32(params.w) {
        output[row * params.w + col] = 0.0;
        return;
    }

    let y0 = i32(floor(src_y));
    let x0 = i32(floor(src_x));
    let y1 = y0 + 1;
    let x1 = x0 + 1;

    let fy = src_y - floor(src_y);
    let fx = src_x - floor(src_x);

    let p00 = sample(y0, x0);
    let p10 = sample(y1, x0);
    let p01 = sample(y0, x1);
    let p11 = sample(y1, x1);

    let top    = p00 * (1.0 - fx) + p01 * fx;
    let bottom = p10 * (1.0 - fx) + p11 * fx;
    let value  = top * (1.0 - fy) + bottom * fy;

    output[row * params.w + col] = value;
}
