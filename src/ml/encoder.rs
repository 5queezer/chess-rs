use crate::board::Board;
use candle_core::{DType, Device, Result, Tensor};

pub const PIECE_PLANES: usize = 12;
pub const HISTORY_POSITIONS: usize = 8;
pub const AUXILIARY_PLANES: usize = 7;
pub const TOTAL_PLANES: usize = PIECE_PLANES * HISTORY_POSITIONS + AUXILIARY_PLANES;

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

    #[allow(dead_code)]
    pub fn push_position(&mut self, board: &Board) {
        self.history.push(board.clone());
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    #[allow(dead_code)]
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    pub fn encode(&self, board: &Board, device: &Device) -> Result<Tensor> {
        let mut planes = vec![0u8; TOTAL_PLANES * 64];

        for (hist_idx, hist_board) in self.history.iter().rev().enumerate() {
            if hist_idx >= HISTORY_POSITIONS {
                break;
            }
            self.encode_pieces(hist_board, &mut planes, hist_idx * PIECE_PLANES * 64);
        }

        let positions_encoded = self.history.len().min(HISTORY_POSITIONS);
        for hist_idx in positions_encoded..HISTORY_POSITIONS {
            self.encode_pieces(board, &mut planes, hist_idx * PIECE_PLANES * 64);
        }

        let aux_offset = PIECE_PLANES * HISTORY_POSITIONS * 64;

        if board.can_castle_kingside(0) {
            for sq in 0..64 {
                planes[aux_offset + sq] = 1;
            }
        }
        if board.can_castle_queenside(0) {
            for sq in 0..64 {
                planes[aux_offset + 64 + sq] = 1;
            }
        }
        if board.can_castle_kingside(1) {
            for sq in 0..64 {
                planes[aux_offset + 128 + sq] = 1;
            }
        }
        if board.can_castle_queenside(1) {
            for sq in 0..64 {
                planes[aux_offset + 192 + sq] = 1;
            }
        }

        if board.side() == 0 {
            for sq in 0..64 {
                planes[aux_offset + 256 + sq] = 1;
            }
        }

        if board.has_en_passant() {
            for sq in 0..64 {
                planes[aux_offset + 320 + sq] = 1;
            }
        }

        let halfmove_normalized = (board.halfmove_clock() as f32 / 50.0).min(1.0);
        let halfmove_u8 = (halfmove_normalized * 255.0) as u8;
        for sq in 0..64 {
            planes[aux_offset + 384 + sq] = halfmove_u8;
        }

        let tensor = Tensor::from_vec(planes, (TOTAL_PLANES, 8, 8), device)?;
        tensor.to_dtype(DType::F32)?.affine(1.0 / 255.0, 0.0)
    }

    fn encode_pieces(&self, board: &Board, planes: &mut [u8], offset: usize) {
        for piece in 0..6 {
            let mut bb = board.piece_bitboard(0, piece);
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                planes[offset + piece * 64 + sq] = 1;
                bb &= bb - 1;
            }

            let mut bb = board.piece_bitboard(1, piece);
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                planes[offset + (piece + 6) * 64 + sq] = 1;
                bb &= bb - 1;
            }
        }
    }

    pub fn encode_perspective(&self, board: &Board, device: &Device) -> Result<Tensor> {
        let tensor = self.encode(board, device)?;

        if board.side() == 1 {
            self.flip_vertical(&tensor)
        } else {
            Ok(tensor)
        }
    }

    fn flip_vertical(&self, tensor: &Tensor) -> Result<Tensor> {
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

        assert_eq!(tensor.dims()[0], TOTAL_PLANES);
    }
}
