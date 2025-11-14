// GPU Detection and Device Management
// Automatically detects and selects the best available compute device

use candle_core::{Device, Result};

/// Device selection configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePreference {
    /// Automatically select best available device
    Auto,
    /// Force CPU usage
    Cpu,
    /// Force CUDA GPU usage (will fail if not available)
    Cuda,
    /// Force Metal GPU usage (will fail if not available)
    Metal,
}

/// Device manager for neural network inference
pub struct DeviceManager {
    device: Device,
    device_type: DeviceType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    Cuda(usize),  // GPU index
    Metal,
}

impl DeviceManager {
    /// Create a new device manager with automatic device selection
    pub fn new() -> Result<Self> {
        Self::with_preference(DevicePreference::Auto)
    }

    /// Create a device manager with specific preference
    pub fn with_preference(pref: DevicePreference) -> Result<Self> {
        let (device, device_type) = match pref {
            DevicePreference::Auto => Self::auto_select()?,
            DevicePreference::Cpu => (Device::Cpu, DeviceType::Cpu),
            DevicePreference::Cuda => {
                let device = Device::new_cuda(0)?;
                (device, DeviceType::Cuda(0))
            }
            DevicePreference::Metal => {
                let device = Device::new_metal(0)?;
                (device, DeviceType::Metal)
            }
        };

        eprintln!("🖥️  ML Device: {}", Self::device_name(&device_type));

        Ok(Self {
            device,
            device_type,
        })
    }

    /// Automatically select the best available device
    fn auto_select() -> Result<(Device, DeviceType)> {
        // Try CUDA first (NVIDIA GPUs)
        #[cfg(feature = "cuda")]
        {
            if let Ok(device) = Device::new_cuda(0) {
                eprintln!("🚀 CUDA GPU detected and enabled");
                return Ok((device, DeviceType::Cuda(0)));
            }
        }

        // Try Metal (Apple Silicon)
        #[cfg(feature = "metal")]
        {
            if let Ok(device) = Device::new_metal(0) {
                eprintln!("🚀 Metal GPU detected and enabled");
                return Ok((device, DeviceType::Metal));
            }
        }

        // Fallback to CPU
        eprintln!("⚠️  No GPU detected, using CPU (slower)");
        eprintln!("💡 Tip: Build with --features cuda or --features metal for GPU acceleration");
        Ok((Device::Cpu, DeviceType::Cpu))
    }

    /// Get a reference to the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the device type
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }

    /// Check if using GPU
    pub fn is_gpu(&self) -> bool {
        !matches!(self.device_type, DeviceType::Cpu)
    }

    /// Get human-readable device name
    fn device_name(dtype: &DeviceType) -> String {
        match dtype {
            DeviceType::Cpu => "CPU".to_string(),
            DeviceType::Cuda(idx) => format!("CUDA GPU #{}", idx),
            DeviceType::Metal => "Metal GPU (Apple Silicon)".to_string(),
        }
    }

    /// Get device info as string
    pub fn info(&self) -> String {
        let name = Self::device_name(&self.device_type);
        let accel = if self.is_gpu() { "✓ GPU Accelerated" } else { "CPU Only" };
        format!("{} - {}", name, accel)
    }

    /// Print device capabilities
    pub fn print_capabilities(&self) {
        eprintln!("\n╔═══════════════════════════════════════╗");
        eprintln!("║     ML Engine Device Information      ║");
        eprintln!("╠═══════════════════════════════════════╣");
        eprintln!("║ Device: {:28} ║", Self::device_name(&self.device_type));
        eprintln!("║ Acceleration: {:23} ║", if self.is_gpu() { "GPU" } else { "CPU" });

        #[cfg(feature = "cuda")]
        if matches!(self.device_type, DeviceType::Cuda(_)) {
            eprintln!("║ CUDA Support: ✓                       ║");
            eprintln!("║ Mixed Precision: ✓ (FP16)             ║");
        }

        #[cfg(feature = "metal")]
        if matches!(self.device_type, DeviceType::Metal) {
            eprintln!("║ Metal Support: ✓                      ║");
            eprintln!("║ Apple Silicon Optimized: ✓            ║");
        }

        if matches!(self.device_type, DeviceType::Cpu) {
            eprintln!("║ Note: GPU acceleration not enabled    ║");
            eprintln!("║ Build with --features cuda/metal     ║");
        }

        eprintln!("╚═══════════════════════════════════════╝\n");
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new().expect("Failed to create device manager")
    }
}

/// Helper function to get the best available device
pub fn get_best_device() -> Result<Device> {
    let manager = DeviceManager::new()?;
    Ok(manager.device)
}

/// Check if CUDA is available
pub fn is_cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0).is_ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Check if Metal is available
pub fn is_metal_available() -> bool {
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0).is_ok()
    }
    #[cfg(not(feature = "metal"))]
    {
        false
    }
}

/// Get device capabilities as a string
pub fn get_device_capabilities() -> String {
    #[allow(unused_mut)]
    let mut caps = vec!["CPU"];

    #[cfg(feature = "cuda")]
    if is_cuda_available() {
        caps.push("CUDA");
    }

    #[cfg(feature = "metal")]
    if is_metal_available() {
        caps.push("Metal");
    }

    caps.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_manager_creation() {
        let manager = DeviceManager::new();
        assert!(manager.is_ok());
    }

    #[test]
    fn test_cpu_device() {
        let manager = DeviceManager::with_preference(DevicePreference::Cpu).unwrap();
        assert_eq!(manager.device_type(), DeviceType::Cpu);
        assert!(!manager.is_gpu());
    }

    #[test]
    fn test_device_capabilities() {
        let caps = get_device_capabilities();
        assert!(caps.contains("CPU"));
    }
}
