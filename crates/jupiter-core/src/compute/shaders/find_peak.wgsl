// Two-pass parallel reduction to find the maximum value and its flat index.
//
// Pass 1 (find_peak_local): each workgroup reduces its portion to a local max,
//   writing (value_as_u32_bits, flat_index) into an intermediate buffer.
// Pass 2 (find_peak_global): a single workgroup reduces the intermediate
//   results to a global maximum, writing [row, col, value_bits] into the
//   output buffer.
//
// Output layout: output[0] = row (u32), output[1] = col (u32),
//                output[2] = bitcast<u32>(value) (f32 reinterpreted).

struct Params {
    total_elements: u32,
    num_workgroups: u32,
}

// --- Pass 1 bindings ---

@group(0) @binding(0) var<storage, read>       input:        array<f32>;
@group(0) @binding(1) var<storage, read_write> intermediate: array<u32>; // pairs: [value_bits, index]
@group(0) @binding(2) var<uniform>             params:       Params;

var<workgroup> shared_val: array<f32, 256>;
var<workgroup> shared_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn find_peak_local(
    @builtin(global_invocation_id)  gid: vec3<u32>,
    @builtin(local_invocation_id)   lid: vec3<u32>,
    @builtin(workgroup_id)          wid: vec3<u32>,
) {
    let tid = lid.x;
    let global_id = gid.x;

    // Load element or -inf sentinel when out of range.
    if global_id < params.total_elements {
        shared_val[tid] = input[global_id];
        shared_idx[tid] = global_id;
    } else {
        shared_val[tid] = -1.0e38;
        shared_idx[tid] = 0u;
    }
    workgroupBarrier();

    // Tree reduction within workgroup.
    var stride = 128u;
    loop {
        if stride == 0u {
            break;
        }
        if tid < stride {
            if shared_val[tid + stride] > shared_val[tid] {
                shared_val[tid] = shared_val[tid + stride];
                shared_idx[tid] = shared_idx[tid + stride];
            }
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    // Thread 0 writes workgroup result.
    if tid == 0u {
        let out_base = wid.x * 2u;
        intermediate[out_base]      = bitcast<u32>(shared_val[0]);
        intermediate[out_base + 1u] = shared_idx[0];
    }
}

// --- Pass 2 bindings ---
// Reuses the same bind group layout; the caller rebinds:
//   binding 0 -> intermediate (as read, reinterpreted)
//   binding 1 -> output (3 x u32: row, col, value_bits)
//   binding 2 -> params (same struct; total_elements ignored, num_workgroups used)

struct Pass2Params {
    total_elements: u32, // unused in pass 2, kept for struct compatibility
    num_workgroups: u32,
}

@group(0) @binding(3) var<storage, read>       inter_in:   array<u32>; // pairs from pass 1
@group(0) @binding(4) var<storage, read_write> result:     array<u32>; // [row, col, value_bits]
@group(0) @binding(5) var<uniform>             params2:    Pass2Params;

var<workgroup> s2_val: array<f32, 256>;
var<workgroup> s2_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn find_peak_global(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;

    // Load from intermediate pairs.
    if tid < params2.num_workgroups {
        let base = tid * 2u;
        s2_val[tid] = bitcast<f32>(inter_in[base]);
        s2_idx[tid] = inter_in[base + 1u];
    } else {
        s2_val[tid] = -1.0e38;
        s2_idx[tid] = 0u;
    }
    workgroupBarrier();

    var stride = 128u;
    loop {
        if stride == 0u {
            break;
        }
        if tid < stride {
            if s2_val[tid + stride] > s2_val[tid] {
                s2_val[tid] = s2_val[tid + stride];
                s2_idx[tid] = s2_idx[tid + stride];
            }
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if tid == 0u {
        let flat = s2_idx[0];
        // We need the image width to recover (row, col). The caller encodes
        // total_elements = h * w, and we receive num_workgroups from pass 1.
        // However, the caller must supply the width separately. We store the
        // flat index and the value so the host can decompose.
        //
        // Convention: result[0] = flat index, result[1] = 0 (reserved),
        //             result[2] = value as u32 bits.
        // The host converts flat -> (row, col) with known width.
        result[0] = flat;
        result[1] = 0u;
        result[2] = bitcast<u32>(s2_val[0]);
    }
}
