// Radix-2 Stockham FFT — out-of-place, one stage per dispatch.
// Data layout: interleaved complex [re, im, re, im, ...].
// The host dispatches one call per log2(n) stage, ping-ponging buf_in / buf_out.
//
// direction =  1.0 → forward DFT
// direction = -1.0 → inverse DFT  (caller must normalize separately)
//
// batch_count  — number of independent 1-D FFTs in the buffer
// batch_stride — distance (in complex elements) between consecutive batches

const TAU: f32 = 6.283185307179586;

struct Params {
    n:            u32, // FFT length (must be power of 2)
    stage:        u32, // current stage index (0-based)
    direction:    f32, // 1.0 forward, -1.0 inverse
    batch_count:  u32,
    batch_stride: u32,
}

@group(0) @binding(0) var<storage, read>       buf_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write> buf_out: array<f32>;
@group(0) @binding(2) var<uniform>             params:  Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_id = gid.x;
    let half_n = params.n >> 1u;
    let total_butterflies = half_n * params.batch_count;

    if thread_id >= total_butterflies {
        return;
    }

    // Identify which batch and which butterfly within the batch.
    let batch   = thread_id / half_n;
    let local_k = thread_id % half_n;

    // Stage parameters.
    let m     = 1u << (params.stage + 1u); // sub-DFT size at this stage
    let half_m = m >> 1u;

    // Butterfly indices within the sub-DFT.
    let group  = local_k / half_m;         // which sub-DFT
    let j      = local_k % half_m;         // position within sub-DFT

    // Input indices (Stockham auto-sort addressing).
    let in_even = group * half_m + j;
    let in_odd  = in_even + half_n;

    // Output indices.
    let out_top    = group * m + j;
    let out_bottom = out_top + half_m;

    // Apply batch offset (in complex-element units, so multiply by 2 for f32 index).
    let batch_off = batch * params.batch_stride * 2u;

    let in_even_ri = batch_off + in_even * 2u;
    let in_odd_ri  = batch_off + in_odd  * 2u;
    let out_top_ri = batch_off + out_top * 2u;
    let out_bot_ri = batch_off + out_bottom * 2u;

    let e_re = buf_in[in_even_ri];
    let e_im = buf_in[in_even_ri + 1u];
    let o_re = buf_in[in_odd_ri];
    let o_im = buf_in[in_odd_ri + 1u];

    // Twiddle factor: W_N^k = exp(-j * direction * TAU * k / N)
    // where k = j at this stage's sub-DFT, and N = m.
    let angle = params.direction * TAU * f32(j) / f32(m);
    let tw_re = cos(angle);
    let tw_im = sin(angle);

    // Complex multiply: twiddle * odd
    let to_re = tw_re * o_re - tw_im * o_im;
    let to_im = tw_re * o_im + tw_im * o_re;

    // Butterfly.
    buf_out[out_top_ri]      = e_re + to_re;
    buf_out[out_top_ri + 1u] = e_im + to_im;
    buf_out[out_bot_ri]      = e_re - to_re;
    buf_out[out_bot_ri + 1u] = e_im - to_im;
}
