// Matrix transpose shader using shared-memory tile to coalesce global memory
// accesses.  output[col * rows + row] = input[row * cols + col].

const TILE_DIM: u32 = 16u;

struct Params {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;

// Tile with one extra column to avoid shared-memory bank conflicts.
var<workgroup> tile: array<array<f32, 17>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
    @builtin(workgroup_id)         wid: vec3<u32>,
) {
    // --- Load phase: read from input into shared tile ---
    let in_col = wid.x * TILE_DIM + lid.x;
    let in_row = wid.y * TILE_DIM + lid.y;

    if in_row < params.rows && in_col < params.cols {
        tile[lid.y][lid.x] = input[in_row * params.cols + in_col];
    }

    workgroupBarrier();

    // --- Store phase: write from shared tile into transposed position ---
    // Swap workgroup x/y so we write coalesced in output.
    let out_col = wid.y * TILE_DIM + lid.x;
    let out_row = wid.x * TILE_DIM + lid.y;

    if out_row < params.cols && out_col < params.rows {
        // Read from tile with swapped local indices to perform the transpose.
        output[out_row * params.rows + out_col] = tile[lid.x][lid.y];
    }
}
