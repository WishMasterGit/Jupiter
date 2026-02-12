//! wgpu-based GPU compute backend (Metal / Vulkan / DX12).

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use ndarray::Array2;
use wgpu::util::DeviceExt;

use super::{BufferInner, ComputeBackend, GpuBuffer};

// ---------------------------------------------------------------------------
// Inline WGSL shaders for small utility operations
// ---------------------------------------------------------------------------

const NORMALIZE_WGSL: &str = r"
struct Params { count: u32, scale: f32 }
@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.count { return; }
    output[gid.x] = input[gid.x] * params.scale;
}
";

const TRANSPOSE_COMPLEX_WGSL: &str = r"
struct Params { rows: u32, cols: u32 }
@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y; let col = gid.x;
    if row >= params.rows || col >= params.cols { return; }
    let i = (row * params.cols + col) * 2u;
    let o = (col * params.rows + row) * 2u;
    output[o] = input[i];
    output[o + 1u] = input[i + 1u];
}
";

const PAD_REAL_TO_COMPLEX_WGSL: &str = r"
struct Params { h: u32, w: u32, padded_h: u32, padded_w: u32 }
@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y; let col = gid.x;
    if row >= params.padded_h || col >= params.padded_w { return; }
    let out_base = (row * params.padded_w + col) * 2u;
    if row < params.h && col < params.w {
        output[out_base] = input[row * params.w + col];
    } else {
        output[out_base] = 0.0;
    }
    output[out_base + 1u] = 0.0;
}
";

const DIVIDE_REAL_WGSL: &str = r"
struct Params { count: u32, epsilon: f32 }
@group(0) @binding(0) var<storage, read>       a:      array<f32>;
@group(0) @binding(1) var<storage, read>       b:      array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform>             params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.count { return; }
    output[gid.x] = a[gid.x] / (b[gid.x] + params.epsilon);
}
";

const MULTIPLY_REAL_WGSL: &str = r"
struct Params { count: u32 }
@group(0) @binding(0) var<storage, read>       a:      array<f32>;
@group(0) @binding(1) var<storage, read>       b:      array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform>             params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.count { return; }
    output[gid.x] = a[gid.x] * b[gid.x];
}
";

const EXTRACT_REAL_SCALED_WGSL: &str = r"
struct Params { out_h: u32, out_w: u32, padded_w: u32, scale: f32 }
@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y; let col = gid.x;
    if row >= params.out_h || col >= params.out_w { return; }
    output[row * params.out_w + col] = input[(row * params.padded_w + col) * 2u] * params.scale;
}
";

// ---------------------------------------------------------------------------
// Uniform parameter structs (must match WGSL layouts exactly)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HannParams {
    h: u32,
    w: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CountParams {
    count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShiftParams {
    h: u32,
    w: u32,
    dx: f32,
    dy: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PeakParams {
    total_elements: u32,
    num_workgroups: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FftParams {
    n: u32,
    stage: u32,
    direction: f32,
    batch_count: u32,
    batch_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TransposeParams {
    rows: u32,
    cols: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ConvolveParams {
    h: u32,
    w: u32,
    kernel_len: u32,
    step: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NormParams {
    count: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PadR2CParams {
    h: u32,
    w: u32,
    padded_h: u32,
    padded_w: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ExtractRealParams {
    out_h: u32,
    out_w: u32,
    padded_w: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DivideRealParams {
    count: u32,
    epsilon: f32,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn gpu_buf(buf: &GpuBuffer) -> &wgpu::Buffer {
    match &buf.inner {
        BufferInner::Wgpu { buffer, .. } => buffer,
        _ => panic!("WgpuBackend: expected GPU buffer"),
    }
}

const fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

// ---------------------------------------------------------------------------
// WgpuBackend
// ---------------------------------------------------------------------------

pub struct WgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter_name: String,
    // Pipelines
    hann_pipeline: wgpu::ComputePipeline,
    cross_power_pipeline: wgpu::ComputePipeline,
    shift_pipeline: wgpu::ComputePipeline,
    peak_local_pipeline: wgpu::ComputePipeline,
    peak_global_pipeline: wgpu::ComputePipeline,
    fft_pipeline: wgpu::ComputePipeline,
    transpose_complex_pipeline: wgpu::ComputePipeline,
    pad_r2c_pipeline: wgpu::ComputePipeline,
    extract_real_pipeline: wgpu::ComputePipeline,
    convolve_rows_pipeline: wgpu::ComputePipeline,
    convolve_cols_pipeline: wgpu::ComputePipeline,
    complex_mul_pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    normalize_pipeline: wgpu::ComputePipeline,
    divide_real_pipeline: wgpu::ComputePipeline,
    multiply_real_pipeline: wgpu::ComputePipeline,
}

impl WgpuBackend {
    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| format!("No suitable GPU adapter found: {e}"))?;

        let adapter_name = adapter.get_info().name.clone();
        tracing::info!("GPU adapter: {adapter_name}");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("jupiter"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
        ))
        .map_err(|e| format!("Failed to create GPU device: {e}"))?;

        let device: Arc<wgpu::Device> = Arc::new(device);
        let queue: Arc<wgpu::Queue> = Arc::new(queue);

        // Compile all shader modules
        let mk = |label, src: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };

        let hann_mod = mk("hann", include_str!("shaders/hann_window.wgsl"));
        let cross_mod = mk("cross_power", include_str!("shaders/cross_power.wgsl"));
        let shift_mod = mk("shift", include_str!("shaders/bilinear_shift.wgsl"));
        let peak_mod = mk("find_peak", include_str!("shaders/find_peak.wgsl"));
        let fft_mod = mk("fft", include_str!("shaders/fft_stockham.wgsl"));
        let conv_mod = mk("convolve", include_str!("shaders/convolve_separable.wgsl"));
        let elem_mod = mk("elementwise", include_str!("shaders/elementwise.wgsl"));
        let norm_mod = mk("normalize", NORMALIZE_WGSL);
        let tc_mod = mk("transpose_complex", TRANSPOSE_COMPLEX_WGSL);
        let pr2c_mod = mk("pad_r2c", PAD_REAL_TO_COMPLEX_WGSL);
        let extr_mod = mk("extract_real", EXTRACT_REAL_SCALED_WGSL);
        let div_mod = mk("divide_real", DIVIDE_REAL_WGSL);
        let mul_mod = mk("multiply_real", MULTIPLY_REAL_WGSL);

        // Create compute pipelines
        let pipe = |module: &wgpu::ShaderModule, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Ok(Self {
            adapter_name,
            hann_pipeline: pipe(&hann_mod, "main"),
            cross_power_pipeline: pipe(&cross_mod, "main"),
            shift_pipeline: pipe(&shift_mod, "main"),
            peak_local_pipeline: pipe(&peak_mod, "find_peak_local"),
            peak_global_pipeline: pipe(&peak_mod, "find_peak_global"),
            fft_pipeline: pipe(&fft_mod, "main"),
            transpose_complex_pipeline: pipe(&tc_mod, "main"),
            pad_r2c_pipeline: pipe(&pr2c_mod, "main"),
            extract_real_pipeline: pipe(&extr_mod, "main"),
            convolve_rows_pipeline: pipe(&conv_mod, "convolve_rows"),
            convolve_cols_pipeline: pipe(&conv_mod, "convolve_cols"),
            complex_mul_pipeline: pipe(&elem_mod, "main"),
            normalize_pipeline: pipe(&norm_mod, "main"),
            divide_real_pipeline: pipe(&div_mod, "main"),
            multiply_real_pipeline: pipe(&mul_mod, "main"),
            device,
            queue,
        })
    }

    // --- Buffer helpers ---

    fn create_storage(&self, data: &[f32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
    }

    fn create_storage_uninit(&self, byte_size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn create_uniform<T: Pod>(&self, data: &T) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(data),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    fn download_f32(&self, buffer: &wgpu::Buffer) -> Vec<f32> {
        let size = buffer.size();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv()
            .expect("GPU channel closed")
            .expect("Buffer mapping failed");

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    fn download_u32(&self, buffer: &wgpu::Buffer) -> Vec<u32> {
        let size = buffer.size();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        rx.recv()
            .expect("GPU channel closed")
            .expect("Buffer mapping failed");

        let data = slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    /// Dispatch a single compute pass with one bind group at group(0).
    fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        entries: &[wgpu::BindGroupEntry],
        workgroups: (u32, u32, u32),
    ) {
        let layout = pipeline.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
        self.queue.submit(std::iter::once(enc.finish()));
    }

    fn make_gpu_buffer(&self, buffer: wgpu::Buffer, height: usize, width: usize) -> GpuBuffer {
        GpuBuffer {
            inner: BufferInner::Wgpu {
                buffer,
                device: Arc::clone(&self.device),
                queue: Arc::clone(&self.queue),
            },
            height,
            width,
        }
    }

    // --- FFT internal helpers ---

    /// Run batch 1-D Stockham FFT stages. Returns the buffer holding the result.
    /// `total_complex` is the total number of complex elements across all batches.
    fn fft_1d_batch(
        &self,
        input: &wgpu::Buffer,
        total_complex: u32,
        n: u32,
        batch_count: u32,
        batch_stride: u32,
        direction: f32,
    ) -> wgpu::Buffer {
        let byte_size = (total_complex as u64) * 2 * 4;
        let buf_a = self.create_storage_uninit(byte_size);
        let buf_b = self.create_storage_uninit(byte_size);
        let num_stages = n.trailing_zeros();

        // Pre-create uniform buffers for each stage
        let stage_uniforms: Vec<wgpu::Buffer> = (0..num_stages)
            .map(|s| {
                self.create_uniform(&FftParams {
                    n,
                    stage: s,
                    direction,
                    batch_count,
                    batch_stride,
                })
            })
            .collect();

        let fft_layout = self.fft_pipeline.get_bind_group_layout(0);
        let butterflies_per_batch = n / 2;
        let total_butterflies = butterflies_per_batch * batch_count;
        let wg_x = div_ceil(total_butterflies, 256);

        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(input, 0, &buf_a, 0, byte_size);

        for stage in 0..num_stages {
            let (src, dst) = if stage % 2 == 0 {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &fft_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: stage_uniforms[stage as usize].as_entire_binding(),
                    },
                ],
            });

            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.fft_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(wg_x, 1, 1);
            }
        }

        self.queue.submit(std::iter::once(enc.finish()));

        // Result is in buf_a for even number of stages, buf_b for odd
        if num_stages % 2 == 0 {
            buf_a
        } else {
            buf_b
        }
    }

    /// Transpose complex matrix from (rows, cols) to (cols, rows).
    /// Buffer holds rows*cols*2 f32 values (interleaved complex).
    fn transpose_complex_buf(&self, input: &wgpu::Buffer, rows: u32, cols: u32) -> wgpu::Buffer {
        let byte_size = (rows as u64) * (cols as u64) * 2 * 4;
        let output = self.create_storage_uninit(byte_size);
        let uniform = self.create_uniform(&TransposeParams { rows, cols });

        self.dispatch(
            &self.transpose_complex_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
            (div_ceil(cols, 16), div_ceil(rows, 16), 1),
        );
        output
    }
}

// ---------------------------------------------------------------------------
// ComputeBackend implementation
// ---------------------------------------------------------------------------

impl ComputeBackend for WgpuBackend {
    fn name(&self) -> &str {
        &self.adapter_name
    }

    fn is_gpu(&self) -> bool {
        true
    }

    fn upload(&self, data: &Array2<f32>) -> GpuBuffer {
        let (h, w) = data.dim();
        let flat: Vec<f32> = data.iter().copied().collect();
        let buffer = self.create_storage(&flat);
        self.make_gpu_buffer(buffer, h, w)
    }

    fn download(&self, buf: &GpuBuffer) -> Array2<f32> {
        let buffer = gpu_buf(buf);
        let num_f32s = (buffer.size() / 4) as usize;
        let storage_w = num_f32s / buf.height;
        let data = self.download_f32(buffer);
        Array2::from_shape_vec((buf.height, storage_w), data).expect("shape mismatch in download")
    }

    fn hann_window(&self, input: &GpuBuffer) -> GpuBuffer {
        let buf = gpu_buf(input);
        let (h, w) = (input.height as u32, input.width as u32);
        let out = self.create_storage_uninit(buf.size());
        let uniform = self.create_uniform(&HannParams { h, w });

        self.dispatch(
            &self.hann_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
            (div_ceil(w, 16), div_ceil(h, 16), 1),
        );
        self.make_gpu_buffer(out, input.height, input.width)
    }

    fn cross_power_spectrum(&self, a: &GpuBuffer, b: &GpuBuffer) -> GpuBuffer {
        let a_buf = gpu_buf(a);
        let b_buf = gpu_buf(b);
        let complex_count = (a.height * a.width) as u32; // logical width = real width
        let byte_size = (complex_count as u64) * 2 * 4;
        let out = self.create_storage_uninit(byte_size);
        let uniform = self.create_uniform(&CountParams {
            count: complex_count,
        });

        self.dispatch(
            &self.cross_power_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform.as_entire_binding(),
                },
            ],
            (div_ceil(complex_count, 256), 1, 1),
        );
        self.make_gpu_buffer(out, a.height, a.width)
    }

    fn find_peak(&self, input: &GpuBuffer) -> (usize, usize, f64) {
        let buf = gpu_buf(input);
        let total = (input.height * input.width) as u32;
        let num_wg = div_ceil(total, 256);

        // Pass 1: per-workgroup reduction
        let intermediate = self.create_storage_uninit((num_wg as u64) * 2 * 4); // pairs of u32
        let params1 = self.create_uniform(&PeakParams {
            total_elements: total,
            num_workgroups: num_wg,
        });

        let layout1 = self.peak_local_pipeline.get_bind_group_layout(0);
        let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: intermediate.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params1.as_entire_binding(),
                },
            ],
        });

        // Pass 2: global reduction
        let result_buf = self.create_storage_uninit(3 * 4); // [flat_idx, 0, value_bits]
        let params2 = self.create_uniform(&PeakParams {
            total_elements: 0,
            num_workgroups: num_wg,
        });

        let layout2 = self.peak_global_pipeline.get_bind_group_layout(0);
        let bg2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: intermediate.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: result_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params2.as_entire_binding(),
                },
            ],
        });

        // Submit both passes
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.peak_local_pipeline);
            pass.set_bind_group(0, &bg1, &[]);
            pass.dispatch_workgroups(num_wg, 1, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.peak_global_pipeline);
            pass.set_bind_group(0, &bg2, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.queue.submit(std::iter::once(enc.finish()));

        // Read back result
        let result = self.download_u32(&result_buf);
        let flat = result[0] as usize;
        let value = f32::from_bits(result[2]) as f64;
        let w = input.width;
        let row = flat / w;
        let col = flat % w;
        (row, col, value)
    }

    fn shift_bilinear(&self, input: &GpuBuffer, dx: f64, dy: f64) -> GpuBuffer {
        let buf = gpu_buf(input);
        let (h, w) = (input.height as u32, input.width as u32);
        let out = self.create_storage_uninit(buf.size());
        let uniform = self.create_uniform(&ShiftParams {
            h,
            w,
            dx: dx as f32,
            dy: dy as f32,
        });

        self.dispatch(
            &self.shift_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
            (div_ceil(w, 16), div_ceil(h, 16), 1),
        );
        self.make_gpu_buffer(out, input.height, input.width)
    }

    fn complex_mul(&self, a: &GpuBuffer, b: &GpuBuffer) -> GpuBuffer {
        let a_buf = gpu_buf(a);
        let b_buf = gpu_buf(b);
        let complex_count = (a.height * a.width) as u32;
        let byte_size = (complex_count as u64) * 2 * 4;
        let out = self.create_storage_uninit(byte_size);
        let uniform = self.create_uniform(&CountParams {
            count: complex_count,
        });

        self.dispatch(
            &self.complex_mul_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform.as_entire_binding(),
                },
            ],
            (div_ceil(complex_count, 256), 1, 1),
        );
        self.make_gpu_buffer(out, a.height, a.width)
    }

    fn divide_real(&self, a: &GpuBuffer, b: &GpuBuffer, epsilon: f32) -> GpuBuffer {
        let a_buf = gpu_buf(a);
        let b_buf = gpu_buf(b);
        let count = (a.height * a.width) as u32;
        let out = self.create_storage_uninit(a_buf.size());
        let uniform = self.create_uniform(&DivideRealParams { count, epsilon });

        self.dispatch(
            &self.divide_real_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform.as_entire_binding(),
                },
            ],
            (div_ceil(count, 256), 1, 1),
        );
        self.make_gpu_buffer(out, a.height, a.width)
    }

    fn multiply_real(&self, a: &GpuBuffer, b: &GpuBuffer) -> GpuBuffer {
        let a_buf = gpu_buf(a);
        let b_buf = gpu_buf(b);
        let count = (a.height * a.width) as u32;
        let out = self.create_storage_uninit(a_buf.size());
        let uniform = self.create_uniform(&CountParams { count });

        self.dispatch(
            &self.multiply_real_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform.as_entire_binding(),
                },
            ],
            (div_ceil(count, 256), 1, 1),
        );
        self.make_gpu_buffer(out, a.height, a.width)
    }

    fn convolve_separable(&self, input: &GpuBuffer, kernel: &[f32]) -> GpuBuffer {
        let buf = gpu_buf(input);
        let (h, w) = (input.height as u32, input.width as u32);
        let total_pixels = h * w;

        let kernel_buf = self.create_storage(kernel);
        let params = self.create_uniform(&ConvolveParams {
            h,
            w,
            kernel_len: kernel.len() as u32,
            step: 1,
        });

        // Row pass
        let after_rows = self.create_storage_uninit(buf.size());
        self.dispatch(
            &self.convolve_rows_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: after_rows.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: kernel_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
            (div_ceil(total_pixels, 256), 1, 1),
        );

        // Column pass
        let after_cols = self.create_storage_uninit(buf.size());
        self.dispatch(
            &self.convolve_cols_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: after_rows.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: after_cols.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: kernel_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
            (div_ceil(total_pixels, 256), 1, 1),
        );

        self.make_gpu_buffer(after_cols, input.height, input.width)
    }

    fn atrous_convolve(&self, input: &GpuBuffer, scale: usize) -> GpuBuffer {
        let buf = gpu_buf(input);
        let (h, w) = (input.height as u32, input.width as u32);
        let total_pixels = h * w;
        let step = 1u32 << scale;

        let b3_kernel: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
        let kernel_buf = self.create_storage(&b3_kernel);
        let params = self.create_uniform(&ConvolveParams {
            h,
            w,
            kernel_len: 5,
            step,
        });

        // Row pass
        let after_rows = self.create_storage_uninit(buf.size());
        self.dispatch(
            &self.convolve_rows_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: after_rows.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: kernel_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
            (div_ceil(total_pixels, 256), 1, 1),
        );

        // Column pass
        let after_cols = self.create_storage_uninit(buf.size());
        self.dispatch(
            &self.convolve_cols_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: after_rows.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: after_cols.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: kernel_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
            (div_ceil(total_pixels, 256), 1, 1),
        );

        self.make_gpu_buffer(after_cols, input.height, input.width)
    }

    fn fft2d(&self, input: &GpuBuffer) -> GpuBuffer {
        let buf = gpu_buf(input);
        let (h, w) = (input.height, input.width);
        let ph = h.next_power_of_two() as u32;
        let pw = w.next_power_of_two() as u32;

        // Step 1: Zero-pad + real-to-complex
        let complex_bytes = (ph as u64) * (pw as u64) * 2 * 4;
        let padded_complex = self.create_storage_uninit(complex_bytes);
        let pad_params = self.create_uniform(&PadR2CParams {
            h: h as u32,
            w: w as u32,
            padded_h: ph,
            padded_w: pw,
        });
        self.dispatch(
            &self.pad_r2c_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: padded_complex.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pad_params.as_entire_binding(),
                },
            ],
            (div_ceil(pw, 16), div_ceil(ph, 16), 1),
        );

        // Step 2: Row-wise FFT
        let total_complex = ph * pw;
        let after_rows = self.fft_1d_batch(
            &padded_complex,
            total_complex,
            pw,
            ph,  // batch_count = number of rows
            pw,  // batch_stride = row length in complex elements
            1.0, // forward
        );

        // Step 3: Transpose (ph, pw) -> (pw, ph)
        let transposed = self.transpose_complex_buf(&after_rows, ph, pw);

        // Step 4: Column-wise FFT (rows of transposed matrix = columns of original)
        let after_cols = self.fft_1d_batch(
            &transposed,
            total_complex,
            ph,
            pw,  // batch_count = number of "rows" in transposed
            ph,  // batch_stride = "row" length in transposed
            1.0, // forward
        );

        // Step 5: Transpose back (pw, ph) -> (ph, pw)
        let result = self.transpose_complex_buf(&after_cols, pw, ph);

        self.make_gpu_buffer(result, ph as usize, pw as usize)
    }

    fn ifft2d_real(&self, input: &GpuBuffer, height: usize, width: usize) -> GpuBuffer {
        let buf = gpu_buf(input);
        // input is (ph, pw) complex interleaved — ph and pw are the padded dims
        let ph = input.height as u32;
        let pw = input.width as u32;
        let total_complex = ph * pw;

        // Step 1: Row-wise IFFT
        let after_rows = self.fft_1d_batch(buf, total_complex, pw, ph, pw, -1.0);

        // Step 2: Transpose (ph, pw) -> (pw, ph)
        let transposed = self.transpose_complex_buf(&after_rows, ph, pw);

        // Step 3: Column-wise IFFT
        let after_cols = self.fft_1d_batch(&transposed, total_complex, ph, pw, ph, -1.0);

        // Step 4: Transpose back (pw, ph) -> (ph, pw)
        let full_result = self.transpose_complex_buf(&after_cols, pw, ph);

        // Step 5: Extract real part, scale by 1/(ph*pw), crop to (height, width)
        let scale = 1.0 / (ph as f32 * pw as f32);
        let out_h = height.min(ph as usize) as u32;
        let out_w = width.min(pw as usize) as u32;
        let out_bytes = (out_h as u64) * (out_w as u64) * 4;
        let output = self.create_storage_uninit(out_bytes);
        let extract_params = self.create_uniform(&ExtractRealParams {
            out_h,
            out_w,
            padded_w: pw,
            scale,
        });

        self.dispatch(
            &self.extract_real_pipeline,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: full_result.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: extract_params.as_entire_binding(),
                },
            ],
            (div_ceil(out_w, 16), div_ceil(out_h, 16), 1),
        );

        self.make_gpu_buffer(output, out_h as usize, out_w as usize)
    }
}
