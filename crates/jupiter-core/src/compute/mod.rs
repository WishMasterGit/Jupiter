mod backend;
pub mod cpu;
#[cfg(feature = "gpu")]
pub mod wgpu_backend;

pub use backend::{create_backend, ComputeBackend, DevicePreference, GpuBuffer};
pub(crate) use backend::BufferInner;
