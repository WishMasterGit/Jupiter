// Real-to-complex conversion shader.
// Copies each real f32 value into the real part of an interleaved complex
// array and sets the imaginary part to 0.0.
// If the output buffer is larger than the input (zero-padding), the extra
// complex elements are filled with (0.0, 0.0).

struct Params {
    count: u32, // number of real input elements
}

@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>; // interleaved [re, im, ...]
@group(0) @binding(2) var<uniform>             params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;

    // Total complex elements is output_size / 2.  We derive it from
    // arrayLength so the shader is self-contained.
    let output_len = arrayLength(&output);
    let num_complex = output_len / 2u;

    if k >= num_complex {
        return;
    }

    let out_ri = k * 2u;

    if k < params.count {
        output[out_ri]      = input[k];
        output[out_ri + 1u] = 0.0;
    } else {
        // Zero-pad region.
        output[out_ri]      = 0.0;
        output[out_ri + 1u] = 0.0;
    }
}
