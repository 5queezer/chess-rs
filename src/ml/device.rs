use candle_core::{Device, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum DevicePreference {
    Auto,
    Cpu,
    Cuda,
    Metal,
}

pub struct DeviceManager {
    device: Device,
    device_type: DeviceType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum DeviceType {
    Cpu,
    Cuda(usize),
    Metal,
}

impl DeviceManager {
    pub fn new() -> Result<Self> {
        Self::with_preference(DevicePreference::Auto)
    }

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

        Ok(Self { device, device_type })
    }

    fn auto_select() -> Result<(Device, DeviceType)> {
        #[cfg(feature = "cuda")]
        {
            if let Ok(device) = Device::new_cuda(0) {
                return Ok((device, DeviceType::Cuda(0)));
            }
        }

        #[cfg(feature = "metal")]
        {
            if let Ok(device) = Device::new_metal(0) {
                return Ok((device, DeviceType::Metal));
            }
        }

        Ok((Device::Cpu, DeviceType::Cpu))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    #[allow(dead_code)]
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }

    pub fn is_gpu(&self) -> bool {
        !matches!(self.device_type, DeviceType::Cpu)
    }

    fn device_name(dtype: &DeviceType) -> String {
        match dtype {
            DeviceType::Cpu => "CPU".to_string(),
            DeviceType::Cuda(idx) => format!("CUDA GPU #{}", idx),
            DeviceType::Metal => "Metal GPU".to_string(),
        }
    }

    pub fn info(&self) -> String {
        let name = Self::device_name(&self.device_type);
        let accel = if self.is_gpu() { "GPU Accelerated" } else { "CPU Only" };
        format!("{} - {}", name, accel)
    }

    #[allow(dead_code)]
    pub fn print_capabilities(&self) {
        eprintln!("Device: {}", Self::device_name(&self.device_type));
        eprintln!("Acceleration: {}", if self.is_gpu() { "GPU" } else { "CPU" });

        #[cfg(feature = "cuda")]
        if matches!(self.device_type, DeviceType::Cuda(_)) {
            eprintln!("CUDA Support: enabled");
        }

        #[cfg(feature = "metal")]
        if matches!(self.device_type, DeviceType::Metal) {
            eprintln!("Metal Support: enabled");
        }
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new().expect("Failed to create device manager")
    }
}

#[allow(dead_code)]
pub fn get_best_device() -> Result<Device> {
    let manager = DeviceManager::new()?;
    Ok(manager.device)
}

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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
