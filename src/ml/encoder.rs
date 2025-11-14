// Neural Network Input Encoder
// Converts chess board state to multi-plane tensor representation
// Compatible with AlphaZero-style networks

use crate::board::Board;
use candle_core::{DType, Device, Result, Tensor};

// Input plane configuration
pub const PIECE_PLANES: usize = 12;  // 6 piece types × 2 colors
pub const HISTORY_POSITIONS: usize = 8;  // Track last 8 positions
pub const AUXILIARY_PLANES: usize = 7;  // Castling, en passant, side, etc.
pub const TOTAL_PLANES: usize = PIECE_PLANES * HISTORY_POSITIONS + AUXILIARY_PLANES;

/// Encodes a chess board into neural network input tensor
/// Shape: [TOTAL_PLANES, 8, 8] = [103, 8, 8]
///
/// Plane layout:
/// - 0-11: Current position (WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK)
/// - 12-23: Position 1 move ago
/// - 24-35: Position 2 moves ago
/// - ... (up to 8 positions of history)
/// - 96: White kingside castling
/// - 97: White queenside castling
/// - 98: Black kingside castling
/// - 99: Black queenside castling
/// - 100: Side to move (1 if white, 0 if black)
/// - 101: En passant available (1 if yes)
/// - 102: Halfmove clock / 50 (normalized)
pub struct BoardEncoder {
    history: Vec<Board>,
    #[allow(dead_code)]
    max_history: usize,
}

impl BoardEncoder {
    pub fn new() -> Self {
        Self {
            history: Vec::with_capacity(HISTORY_POSITIONS),
            max_history: HISTORY_POSITIONS,
        }
    }

    /// Update history with current board position
    #[allow(dead_code)]
    pub fn push_position(&mut self, board: &Board) {
        self.history.push(board.clone());
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Clear position history
    #[allow(dead_code)]
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Encode current board state to tensor
    pub fn encode(&self, board: &Board, device: &Device) -> Result<Tensor> {
        let mut planes = vec![0u8; TOTAL_PLANES * 64];

        // Encode piece positions for current and historical positions
        for (hist_idx, hist_board) in self.history.iter().rev().enumerate() {
            if hist_idx >= HISTORY_POSITIONS {
                break;
            }
            self.encode_pieces(hist_board, &mut planes, hist_idx * PIECE_PLANES * 64);
        }

        // If we have fewer than HISTORY_POSITIONS, fill remaining with current position
        let positions_encoded = self.history.len().min(HISTORY_POSITIONS);
        for hist_idx in positions_encoded..HISTORY_POSITIONS {
            self.encode_pieces(board, &mut planes, hist_idx * PIECE_PLANES * 64);
        }

        // Encode auxiliary features starting at plane 96
        let aux_offset = PIECE_PLANES * HISTORY_POSITIONS * 64;

        // Castling rights (planes 96-99)
        if board.can_castle_kingside(0) {  // White kingside
            for sq in 0..64 {
                planes[aux_offset + sq] = 1;
            }
        }
        if board.can_castle_queenside(0) {  // White queenside
            for sq in 0..64 {
                planes[aux_offset + 64 + sq] = 1;
            }
        }
        if board.can_castle_kingside(1) {  // Black kingside
            for sq in 0..64 {
                planes[aux_offset + 128 + sq] = 1;
            }
        }
        if board.can_castle_queenside(1) {  // Black queenside
            for sq in 0..64 {
                planes[aux_offset + 192 + sq] = 1;
            }
        }

        // Side to move (plane 100)
        if board.side() == 0 {  // White to move
            for sq in 0..64 {
                planes[aux_offset + 256 + sq] = 1;
            }
        }

        // En passant availability (plane 101)
        if board.has_en_passant() {
            for sq in 0..64 {
                planes[aux_offset + 320 + sq] = 1;
            }
        }

        // Halfmove clock normalized (plane 102)
        let halfmove_normalized = (board.halfmove_clock() as f32 / 50.0).min(1.0);
        let halfmove_u8 = (halfmove_normalized * 255.0) as u8;
        for sq in 0..64 {
            planes[aux_offset + 384 + sq] = halfmove_u8;
        }

        // Convert to tensor and reshape to [TOTAL_PLANES, 8, 8]
        let tensor = Tensor::from_vec(planes, (TOTAL_PLANES, 8, 8), device)?;

        // Convert to float32 and normalize to [0, 1]
        tensor.to_dtype(DType::F32)?.affine(1.0 / 255.0, 0.0)
    }

    /// Encode piece positions for a single board state
    fn encode_pieces(&self, board: &Board, planes: &mut [u8], offset: usize) {
        // Piece order: Pawn, Knight, Bishop, Rook, Queen, King
        for piece in 0..6 {
            // White pieces
            let mut bb = board.piece_bitboard(0, piece);
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                planes[offset + piece * 64 + sq] = 1;
                bb &= bb - 1;  // Clear LSB
            }

            // Black pieces
            let mut bb = board.piece_bitboard(1, piece);
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                planes[offset + (piece + 6) * 64 + sq] = 1;
                bb &= bb - 1;  // Clear LSB
            }
        }
    }

    /// Encode board from side to move perspective (flip for black)
    pub fn encode_perspective(&self, board: &Board, device: &Device) -> Result<Tensor> {
        let tensor = self.encode(board, device)?;

        if board.side() == 1 {  // Black to move - flip the board
            // Flip vertically (rank 1 ↔ rank 8)
            self.flip_vertical(&tensor)
        } else {
            Ok(tensor)
        }
    }

    /// Flip tensor vertically (for black's perspective)
    fn flip_vertical(&self, tensor: &Tensor) -> Result<Tensor> {
        // Reverse the middle dimension (8 ranks)
        let shape = tensor.dims();
        let indices: Vec<u32> = (0..shape[1] as u32).rev().collect();
        tensor.index_select(&Tensor::from_vec(indices, shape[1], tensor.device())?, 1)
    }
}

impl Default for BoardEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_shape() {
        let encoder = BoardEncoder::new();
        let board = Board::new();
        let device = Device::Cpu;

        let tensor = encoder.encode(&board, &device).unwrap();
        assert_eq!(tensor.dims(), &[TOTAL_PLANES, 8, 8]);
    }

    #[test]
    fn test_initial_position_encoding() {
        let mut encoder = BoardEncoder::new();
        let board = Board::new();
        encoder.push_position(&board);

        let device = Device::Cpu;
        let tensor = encoder.encode(&board, &device).unwrap();

        // Check that we have the correct number of planes
        assert_eq!(tensor.dims()[0], TOTAL_PLANES);
    }
}
