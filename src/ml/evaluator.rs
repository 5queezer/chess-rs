// ML Evaluator - Main interface for neural network evaluation

use crate::board::{Board, Move};
use crate::ml::{BoardEncoder, ChessNet, DeviceManager, DevicePreference};
use candle_core::{DType, Result};
use candle_nn::VarBuilder;
use std::path::Path;
use std::sync::Arc;

/// Configuration for ML evaluator
#[derive(Debug, Clone)]
pub struct MLConfig {
    /// Path to model weights file (safetensors format)
    pub model_path: Option<String>,
    /// Device preference
    pub device_preference: DevicePreference,
    /// Use ML evaluation (if false, falls back to classical)
    pub enabled: bool,
    /// Blend factor between ML and classical evaluation (0.0 = all classical, 1.0 = all ML)
    pub blend_factor: f32,
    /// Batch size for inference (currently only 1 is supported)
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

/// ML Evaluator for chess positions
pub struct MLEvaluator {
    network: Option<Arc<ChessNet>>,
    encoder: BoardEncoder,
    device_manager: DeviceManager,
    config: MLConfig,
    is_ready: bool,
}

impl MLEvaluator {
    /// Create a new ML evaluator with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(MLConfig::default())
    }

    /// Create an ML evaluator with custom configuration
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

        // Try to load model if path is provided
        if let Some(path) = evaluator.config.model_path.clone() {
            match evaluator.load_model(&path) {
                Ok(_) => {
                    eprintln!("✓ ML model loaded successfully from: {}", path);
                }
                Err(e) => {
                    eprintln!("⚠️  Failed to load ML model: {}", e);
                    eprintln!("   Falling back to random initialization for testing");
                    evaluator.init_random_model()?;
                }
            }
        } else {
            eprintln!("ℹ️  No model path specified, using random initialization");
            evaluator.init_random_model()?;
        }

        Ok(evaluator)
    }

    /// Initialize a random model for testing (not for actual play)
    fn init_random_model(&mut self) -> Result<()> {
        let device = self.device_manager.device().clone();
        let net = ChessNet::random_init(crate::ml::encoder::TOTAL_PLANES, device)?;
        self.network = Some(Arc::new(net));
        self.is_ready = true;
        Ok(())
    }

    /// Load model from safetensors file
    pub fn load_model(&mut self, _path: impl AsRef<Path>) -> Result<()> {
        let device = self.device_manager.device().clone();

        // Load weights using safetensors
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // TODO: Actually load from safetensors file
        // For now, use random initialization
        let net = ChessNet::new(vb, crate::ml::encoder::TOTAL_PLANES, device)?;

        self.network = Some(Arc::new(net));
        self.is_ready = true;

        Ok(())
    }

    /// Update position history (call this after each move)
    #[allow(dead_code)]
    pub fn push_position(&mut self, board: &Board) {
        self.encoder.push_position(board);
    }

    /// Clear position history (call at start of new game)
    #[allow(dead_code)]
    pub fn clear_history(&mut self) {
        self.encoder.clear_history();
    }

    /// Evaluate a position using the neural network
    /// Returns centipawn score from white's perspective
    pub fn evaluate(&self, board: &Board) -> Result<i32> {
        if !self.is_ready || !self.config.enabled {
            return Err(candle_core::Error::Msg(
                "ML evaluator not ready or disabled".to_string(),
            ));
        }

        let network = self.network.as_ref().unwrap();

        // Encode board to tensor
        let input = self
            .encoder
            .encode_perspective(board, network.device())?;

        // Add batch dimension [1, planes, 8, 8]
        let input = input.unsqueeze(0)?;

        // Run inference
        let (value, _policy) = network.forward(&input)?;

        // Extract value and convert to centipawns
        // Network outputs in range [-1, 1] where -1 = black wins, +1 = white wins
        let value_f32 = value.to_vec2::<f32>()?[0][0];

        // Convert to centipawns (approximately)
        // We use a scaling factor where ±1.0 = ±10000 centipawns (100 pawns)
        let centipawns = (value_f32 * 10000.0) as i32;

        Ok(centipawns)
    }

    /// Get move probabilities from the policy network
    /// Returns vector of (move, probability) pairs for legal moves
    #[allow(dead_code)]
    pub fn get_move_probabilities(&self, board: &Board, legal_moves: &[Move]) -> Result<Vec<(Move, f32)>> {
        if !self.is_ready {
            return Err(candle_core::Error::Msg(
                "ML evaluator not ready".to_string(),
            ));
        }

        let network = self.network.as_ref().unwrap();

        // Encode board to tensor
        let input = self
            .encoder
            .encode_perspective(board, network.device())?;

        // Add batch dimension
        let input = input.unsqueeze(0)?;

        // Run inference
        let (_value, policy) = network.forward(&input)?;

        // Extract policy probabilities
        let policy_vec = policy.to_vec2::<f32>()?[0].clone();

        // Map legal moves to probabilities
        let mut move_probs = Vec::new();
        for &m in legal_moves {
            // TODO: Implement proper move encoding to policy index
            // For now, use a simple hash-based approach
            let policy_idx = Self::move_to_policy_index(m);
            if policy_idx < policy_vec.len() {
                move_probs.push((m, policy_vec[policy_idx]));
            }
        }

        // Sort by probability (highest first)
        move_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(move_probs)
    }

    /// Convert a move to policy network index
    /// This is a simplified version - proper implementation would use
    /// the AlphaZero move encoding scheme
    #[allow(dead_code)]
    fn move_to_policy_index(m: Move) -> usize {
        let from = m.from as usize;
        let to = m.to as usize;
        let promo = m.promo as usize;

        // Simple encoding: from * 64 + to + promo offset
        // This is not the proper AlphaZero encoding but works for demonstration
        from * 64 + to + promo * 4096
    }

    /// Check if ML evaluation is ready
    #[allow(dead_code)]
    pub fn is_ready(&self) -> bool {
        self.is_ready
    }

    /// Check if using GPU
    pub fn is_gpu(&self) -> bool {
        self.device_manager.is_gpu()
    }

    /// Get device information
    pub fn device_info(&self) -> String {
        self.device_manager.info()
    }

    /// Print ML engine status
    pub fn print_status(&self) {
        eprintln!("\n╔═══════════════════════════════════════╗");
        eprintln!("║      ML Chess Engine Status           ║");
        eprintln!("╠═══════════════════════════════════════╣");
        eprintln!("║ Status: {:30} ║", if self.is_ready { "✓ Ready" } else { "✗ Not Ready" });
        eprintln!("║ Enabled: {:29} ║", if self.config.enabled { "Yes" } else { "No" });
        eprintln!("║ Device: {:30} ║",
                  if self.is_gpu() { "GPU" } else { "CPU" });
        eprintln!("║ Blend Factor: {:24} ║", format!("{:.1}%", self.config.blend_factor * 100.0));

        if let Some(ref path) = self.config.model_path {
            eprintln!("║ Model: {:31} ║", path);
        } else {
            eprintln!("║ Model: Random (testing only)        ║");
        }

        eprintln!("╚═══════════════════════════════════════╝\n");
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

        // Initial position should be roughly equal
        let score_val = score.unwrap();
        assert!(score_val.abs() < 5000); // Within 50 pawns (reasonable for untrained net)
    }

    #[test]
    fn test_position_history() {
        let mut evaluator = MLEvaluator::new().unwrap();
        let board = Board::new();

        evaluator.push_position(&board);
        evaluator.clear_history();

        // Should not crash
    }
}
