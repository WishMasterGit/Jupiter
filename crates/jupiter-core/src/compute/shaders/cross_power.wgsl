// Cross-power spectrum shader.
// For each complex element: compute a * conj(b), then normalize by magnitude.
// Data layout: interleaved complex [re, im, re, im, ...], so element k is at
// indices 2*k (real) and 2*k+1 (imaginary).

struct Params {
    count: u32, // number of complex elements
}

@group(0) @binding(0) var<storage, read>       a_complex: array<f32>;
@group(0) @binding(1) var<storage, read>       b_complex: array<f32>;
@group(0) @binding(2) var<storage, read_write> output:    array<f32>;
@group(0) @binding(3) var<uniform>             params:    Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= params.count {
        return;
    }

    let ri = k * 2u;
    let ii = ri + 1u;

    let a_re = a_complex[ri];
    let a_im = a_complex[ii];
    let b_re = b_complex[ri];
    let b_im = b_complex[ii];

    // a * conj(b)
    let cross_re = a_re * b_re + a_im * b_im;
    let cross_im = a_im * b_re - a_re * b_im;

    let mag = sqrt(cross_re * cross_re + cross_im * cross_im);
    let inv_mag = select(0.0, 1.0 / mag, mag > 0.0);

    output[ri] = cross_re * inv_mag;
    output[ii] = cross_im * inv_mag;
}
