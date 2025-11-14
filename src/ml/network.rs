// AlphaZero-style Neural Network Architecture
// ResNet with Value and Policy heads

use candle_core::{DType, Device, Module, ModuleT, Result, Tensor};
use candle_nn::{
    batch_norm, conv2d, linear, ops, BatchNorm, Conv2d, Conv2dConfig, Linear, VarBuilder,
};

const NUM_RESIDUAL_BLOCKS: usize = 20;
const NUM_FILTERS: usize = 256;
const POLICY_OUTPUT_SIZE: usize = 1858;  // All possible chess moves

/// Residual Block with skip connection
#[derive(Debug)]
pub struct ResidualBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
}

impl ResidualBlock {
    pub fn new(vb: VarBuilder, channels: usize) -> Result<Self> {
        let conv_config = Conv2dConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        };

        Ok(Self {
            conv1: conv2d(channels, channels, 3, conv_config, vb.pp("conv1"))?,
            bn1: batch_norm(channels, 1e-5, vb.pp("bn1"))?,
            conv2: conv2d(channels, channels, 3, conv_config, vb.pp("conv2"))?,
            bn2: batch_norm(channels, 1e-5, vb.pp("bn2"))?,
        })
    }
}

impl Module for ResidualBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // First convolution
        let out = self.conv1.forward(xs)?;
        let out = self.bn1.forward_t(&out, false)?;  // false = inference mode
        let out = out.relu()?;

        // Second convolution
        let out = self.conv2.forward(&out)?;
        let out = self.bn2.forward_t(&out, false)?;  // false = inference mode

        // Skip connection + ReLU
        let out = (out + xs)?;
        out.relu()
    }
}

/// Value Head - outputs position evaluation
#[derive(Debug)]
pub struct ValueHead {
    conv: Conv2d,
    bn: BatchNorm,
    fc1: Linear,
    fc2: Linear,
}

impl ValueHead {
    pub fn new(vb: VarBuilder, in_channels: usize) -> Result<Self> {
        let conv_config = Conv2dConfig {
            padding: 0,
            stride: 1,
            ..Default::default()
        };

        Ok(Self {
            conv: conv2d(in_channels, 1, 1, conv_config, vb.pp("conv"))?,
            bn: batch_norm(1, 1e-5, vb.pp("bn"))?,
            fc1: linear(64, 256, vb.pp("fc1"))?,  // 8x8 board flattened
            fc2: linear(256, 1, vb.pp("fc2"))?,
        })
    }
}

impl Module for ValueHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = self.conv.forward(xs)?;
        let out = self.bn.forward_t(&out, false)?;  // false = inference mode
        let out = out.relu()?;

        // Flatten spatial dimensions
        let batch_size = out.dims()[0];
        let out = out.reshape((batch_size, 64))?;

        let out = self.fc1.forward(&out)?;
        let out = out.relu()?;

        let out = self.fc2.forward(&out)?;

        // Tanh activation to output in range [-1, 1]
        out.tanh()
    }
}

/// Policy Head - outputs move probabilities
#[derive(Debug)]
pub struct PolicyHead {
    conv: Conv2d,
    bn: BatchNorm,
    fc: Linear,
}

impl PolicyHead {
    pub fn new(vb: VarBuilder, in_channels: usize) -> Result<Self> {
        let conv_config = Conv2dConfig {
            padding: 0,
            stride: 1,
            ..Default::default()
        };

        Ok(Self {
            conv: conv2d(in_channels, 2, 1, conv_config, vb.pp("conv"))?,
            bn: batch_norm(2, 1e-5, vb.pp("bn"))?,
            fc: linear(128, POLICY_OUTPUT_SIZE, vb.pp("fc"))?,  // 2 * 8 * 8 = 128
        })
    }
}

impl Module for PolicyHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = self.conv.forward(xs)?;
        let out = self.bn.forward_t(&out, false)?;  // false = inference mode
        let out = out.relu()?;

        // Flatten spatial dimensions
        let batch_size = out.dims()[0];
        let out = out.reshape((batch_size, 128))?;

        let out = self.fc.forward(&out)?;

        // Softmax to get probabilities
        ops::softmax(&out, 1)
    }
}

/// Main AlphaZero-style Neural Network
#[derive(Debug)]
pub struct ChessNet {
    // Initial convolution
    conv_init: Conv2d,
    bn_init: BatchNorm,

    // Residual tower
    res_blocks: Vec<ResidualBlock>,

    // Output heads
    value_head: ValueHead,
    policy_head: PolicyHead,

    device: Device,
}

impl ChessNet {
    /// Create a new ChessNet with the specified number of input planes
    pub fn new(vb: VarBuilder, input_planes: usize, device: Device) -> Result<Self> {
        let conv_config = Conv2dConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        };

        // Initial convolution to expand from input planes to NUM_FILTERS
        let conv_init = conv2d(
            input_planes,
            NUM_FILTERS,
            3,
            conv_config,
            vb.pp("conv_init"),
        )?;
        let bn_init = batch_norm(NUM_FILTERS, 1e-5, vb.pp("bn_init"))?;

        // Create residual blocks
        let mut res_blocks = Vec::with_capacity(NUM_RESIDUAL_BLOCKS);
        for i in 0..NUM_RESIDUAL_BLOCKS {
            res_blocks.push(ResidualBlock::new(
                vb.pp(format!("res_block_{}", i)),
                NUM_FILTERS,
            )?);
        }

        // Create output heads
        let value_head = ValueHead::new(vb.pp("value_head"), NUM_FILTERS)?;
        let policy_head = PolicyHead::new(vb.pp("policy_head"), NUM_FILTERS)?;

        Ok(Self {
            conv_init,
            bn_init,
            res_blocks,
            value_head,
            policy_head,
            device,
        })
    }

    /// Forward pass through the network
    /// Returns (value, policy) where:
    /// - value: [batch_size, 1] - position evaluation in [-1, 1]
    /// - policy: [batch_size, POLICY_OUTPUT_SIZE] - move probabilities
    pub fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        // Initial convolution
        let mut out = self.conv_init.forward(xs)?;
        out = self.bn_init.forward_t(&out, false)?;  // false = inference mode
        out = out.relu()?;

        // Residual tower
        for block in &self.res_blocks {
            out = block.forward(&out)?;
        }

        // Split into value and policy heads
        let value = self.value_head.forward(&out)?;
        let policy = self.policy_head.forward(&out)?;

        Ok((value, policy))
    }

    /// Evaluate a single position
    /// Returns (value_score, policy_logits)
    #[allow(dead_code)]
    pub fn evaluate_position(&self, input: &Tensor) -> Result<(f32, Vec<f32>)> {
        let (value, policy) = self.forward(input)?;

        // Extract value as scalar
        let value_scalar = value.to_vec1::<f32>()?[0];

        // Extract policy as vector
        let policy_vec = policy.to_vec2::<f32>()?;
        let policy_probs = policy_vec[0].clone();

        Ok((value_scalar, policy_probs))
    }

    /// Get the device this network is on
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Create a simple random-initialized network for testing
    pub fn random_init(input_planes: usize, device: Device) -> Result<Self> {
        // Create dummy variable builder
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        Self::new(vb, input_planes, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let device = Device::Cpu;
        let net = ChessNet::random_init(103, device.clone()).unwrap();

        // Create dummy input
        let input = Tensor::zeros((1, 103, 8, 8), DType::F32, &device).unwrap();

        let (value, policy) = net.forward(&input).unwrap();

        assert_eq!(value.dims(), &[1, 1]);
        assert_eq!(policy.dims(), &[1, POLICY_OUTPUT_SIZE]);
    }

    #[test]
    fn test_residual_block() {
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let block = ResidualBlock::new(vb, NUM_FILTERS).unwrap();
        let input = Tensor::zeros((1, NUM_FILTERS, 8, 8), DType::F32, &device).unwrap();

        let output = block.forward(&input).unwrap();
        assert_eq!(output.dims(), input.dims());
    }
}
