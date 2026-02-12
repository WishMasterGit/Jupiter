// Element-wise complex multiplication of two interleaved complex buffers.
//
// Data layout: interleaved [re, im, re, im, ...], so element k is at
// indices 2*k (real) and 2*k+1 (imaginary).

struct MulParams {
    count: u32, // number of complex elements
}

@group(0) @binding(0) var<storage, read>       a:          array<f32>;
@group(0) @binding(1) var<storage, read>       b:          array<f32>;
@group(0) @binding(2) var<storage, read_write> out_mul:    array<f32>;
@group(0) @binding(3) var<uniform>             mul_params: MulParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= mul_params.count {
        return;
    }

    let ri = k * 2u;
    let ii = ri + 1u;

    let a_re = a[ri];
    let a_im = a[ii];
    let b_re = b[ri];
    let b_im = b[ii];

    out_mul[ri] = a_re * b_re - a_im * b_im;
    out_mul[ii] = a_re * b_im + a_im * b_re;
}
