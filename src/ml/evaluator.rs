use crate::board::{Board, Move};
use crate::ml::{BoardEncoder, ChessNet, DeviceManager, DevicePreference};
use candle_core::{DType, Result};
use candle_nn::VarBuilder;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct MLConfig {
    pub model_path: Option<String>,
    pub device_preference: DevicePreference,
    pub enabled: bool,
    pub blend_factor: f32,
    #[allow(dead_code)]
    pub batch_size: usize,
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            device_preference: DevicePreference::Auto,
            enabled: true,
            blend_factor: 1.0,
            batch_size: 1,
        }
    }
}

pub struct MLEvaluator {
    network: Option<Arc<ChessNet>>,
    encoder: BoardEncoder,
    device_manager: DeviceManager,
    config: MLConfig,
    is_ready: bool,
}

impl MLEvaluator {
    pub fn new() -> Result<Self> {
        Self::with_config(MLConfig::default())
    }

    pub fn with_config(config: MLConfig) -> Result<Self> {
        let device_manager = DeviceManager::with_preference(config.device_preference)?;
        let encoder = BoardEncoder::new();

        let mut evaluator = Self {
            network: None,
            encoder,
            device_manager,
            config,
            is_ready: false,
        };

        if let Some(path) = evaluator.config.model_path.clone() {
            if evaluator.load_model(&path).is_err() {
                evaluator.init_random_model()?;
            }
        } else {
            evaluator.init_random_model()?;
        }

        Ok(evaluator)
    }

    fn init_random_model(&mut self) -> Result<()> {
        let device = self.device_manager.device().clone();
        let net = ChessNet::random_init(crate::ml::encoder::TOTAL_PLANES, device)?;
        self.network = Some(Arc::new(net));
        self.is_ready = true;
        Ok(())
    }

    pub fn load_model(&mut self, _path: impl AsRef<Path>) -> Result<()> {
        let device = self.device_manager.device().clone();
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let net = ChessNet::new(vb, crate::ml::encoder::TOTAL_PLANES, device)?;
        self.network = Some(Arc::new(net));
        self.is_ready = true;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn push_position(&mut self, board: &Board) {
        self.encoder.push_position(board);
    }

    #[allow(dead_code)]
    pub fn clear_history(&mut self) {
        self.encoder.clear_history();
    }

    pub fn evaluate(&self, board: &Board) -> Result<i32> {
        if !self.is_ready || !self.config.enabled {
            return Err(candle_core::Error::Msg("ML evaluator not ready or disabled".to_string()));
        }

        let network = self.network.as_ref().unwrap();
        let input = self.encoder.encode_perspective(board, network.device())?;
        let input = input.unsqueeze(0)?;
        let (value, _policy) = network.forward(&input)?;
        let value_f32 = value.to_vec2::<f32>()?[0][0];
        let centipawns = (value_f32 * 10000.0) as i32;
        Ok(centipawns)
    }

    #[allow(dead_code)]
    pub fn get_move_probabilities(&self, board: &Board, legal_moves: &[Move]) -> Result<Vec<(Move, f32)>> {
        if !self.is_ready {
            return Err(candle_core::Error::Msg("ML evaluator not ready".to_string()));
        }

        let network = self.network.as_ref().unwrap();
        let input = self.encoder.encode_perspective(board, network.device())?;
        let input = input.unsqueeze(0)?;
        let (_value, policy) = network.forward(&input)?;
        let policy_vec = policy.to_vec2::<f32>()?[0].clone();

        let mut move_probs = Vec::new();
        for &m in legal_moves {
            let policy_idx = Self::move_to_policy_index(m);
            if policy_idx < policy_vec.len() {
                move_probs.push((m, policy_vec[policy_idx]));
            }
        }

        move_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(move_probs)
    }

    #[allow(dead_code)]
    fn move_to_policy_index(m: Move) -> usize {
        let from = m.from as usize;
        let to = m.to as usize;
        let promo = m.promo as usize;
        from * 64 + to + promo * 4096
    }

    #[allow(dead_code)]
    pub fn is_ready(&self) -> bool {
        self.is_ready
    }

    pub fn is_gpu(&self) -> bool {
        self.device_manager.is_gpu()
    }

    pub fn device_info(&self) -> String {
        self.device_manager.info()
    }

    pub fn blend_factor(&self) -> f32 {
        self.config.blend_factor
    }

    pub fn set_blend_factor(&mut self, factor: f32) {
        self.config.blend_factor = factor.clamp(0.0, 1.0);
    }

    pub fn print_status(&self) {
        eprintln!("ML Status: {}", if self.is_ready { "Ready" } else { "Not Ready" });
        eprintln!("Device: {}", self.device_info());
        eprintln!("Blend Factor: {}", self.config.blend_factor);
    }
}

impl Default for MLEvaluator {
    fn default() -> Self {
        Self::new().expect("Failed to create ML evaluator")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_evaluator_creation() {
        let evaluator = MLEvaluator::new();
        assert!(evaluator.is_ok());
    }

    #[test]
    fn test_evaluation() {
        let evaluator = MLEvaluator::new().unwrap();
        let board = Board::new();
        let score = evaluator.evaluate(&board);
        assert!(score.is_ok());
        let score_val = score.unwrap();
        assert!(score_val.abs() <= 10000);
    }

    #[test]
    fn test_position_history() {
        let mut evaluator = MLEvaluator::new().unwrap();
        let board = Board::new();
        evaluator.push_position(&board);
        evaluator.clear_history();
    }
}
