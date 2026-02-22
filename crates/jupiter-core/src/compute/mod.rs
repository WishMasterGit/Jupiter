mod backend;
pub mod cpu;
#[cfg(feature = "gpu")]
pub mod wgpu_backend;

pub(crate) use backend::BufferInner;
pub use backend::{create_backend, ComputeBackend, DevicePreference, GpuBuffer};
