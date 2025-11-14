// Machine Learning Module for Chess Engine
// Provides neural network evaluation and policy guidance

pub mod device;
pub mod encoder;
pub mod network;
pub mod evaluator;

pub use device::{DeviceManager, DevicePreference};
pub use encoder::BoardEncoder;
pub use network::ChessNet;
pub use evaluator::MLEvaluator;

#[allow(unused_imports)]
pub use evaluator::MLConfig;
