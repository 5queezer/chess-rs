pub const PAWN: usize = 0;
pub const KNIGHT: usize = 1;
pub const BISHOP: usize = 2;
pub const ROOK: usize = 3;
pub const QUEEN: usize = 4;
pub const KING: usize = 5;

pub const FLAG_CAPTURE: u8 = 1;
pub const FLAG_EN_PASSANT: u8 = 2;
pub const FLAG_CASTLE: u8 = 4;

const FILE_A: u64 = 0x0101010101010101;
const FILE_H: u64 = 0x8080808080808080;
const RANK_3: u64 = 0x0000000000FF0000;
const RANK_6: u64 = 0x0000FF0000000000;
const PROMOTION_PIECES: [u8; 4] = [KNIGHT as u8, BISHOP as u8, ROOK as u8, QUEEN as u8];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    White,
    Black,
}

impl Side {
    pub fn flip(&self) -> Self {
        match self {
            Side::White => Side::Black,
            Side::Black => Side::White,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move {
    pub from: u8,
    pub to: u8,
    pub promo: u8,
    pub flags: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct Undo {
    pub m: Move,
    pub captured: u8,
    pub castling: u8,
    pub ep: u8,
    pub half_move: u8,
    pub hash: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct NullMoveState {
    pub ep: u8,
    pub half_move: u8,
    pub full_move: u16,
    pub hash: u64,
}

#[derive(Clone)]
pub struct Board {
    pub bb_piece: [[u64; 6]; 2],
    pub bb_side: [u64; 2],
    pub stm: Side,
    pub ep: u8,
    pub castling: u8,
    pub half_move: u8,
    pub full_move: u16,
    pub hash: u64,
    pub hist: Vec<Undo>,
}

impl Board {
    pub fn piece_at(&self, side: Side, sq: usize) -> Option<usize> {
        (0..6).find(|&p| self.bb_piece[side as usize][p] & (1u64 << sq) != 0)
    }

    pub fn startpos() -> Self {
        Self::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    }

    pub fn from_fen(fen: &str) -> Self {
        let mut board = Self {
            bb_piece: [[0; 6]; 2],
            bb_side: [0; 2],
            stm: Side::White,
            ep: 255,
            castling: 0,
            half_move: 0,
            full_move: 1,
            hash: 0,
            hist: Vec::new(),
        };

        let mut it = fen.split_whitespace();
        let board_str = it.next().unwrap();
        let mut rank = 7;
        let mut file = 0;

        for c in board_str.chars() {
            if c.is_ascii_digit() {
                file += c.to_digit(10).unwrap();
            } else if c == '/' {
                rank -= 1;
                file = 0;
            } else {
                let side = if c.is_uppercase() { Side::White } else { Side::Black };
                let piece = match c.to_ascii_lowercase() {
                    'p' => PAWN,
                    'n' => KNIGHT,
                    'b' => BISHOP,
                    'r' => ROOK,
                    'q' => QUEEN,
                    'k' => KING,
                    _ => panic!("invalid piece"),
                };
                let sq = rank * 8 + file;
                board.bb_piece[side as usize][piece] |= 1u64 << sq;
                board.bb_side[side as usize] |= 1u64 << sq;
                file += 1;
            }
        }

        board.stm = if it.next().unwrap() == "w" { Side::White } else { Side::Black };

        let castling_str = it.next().unwrap();
        if castling_str.contains('K') { board.castling |= 1; }
        if castling_str.contains('Q') { board.castling |= 2; }
        if castling_str.contains('k') { board.castling |= 4; }
        if castling_str.contains('q') { board.castling |= 8; }

        let ep_str = it.next().unwrap();
        if ep_str != "-" {
            board.ep = parse_square(ep_str).unwrap() as u8;
        }

        board.half_move = it.next().unwrap().parse().unwrap();
        board.full_move = it.next().unwrap().parse().unwrap();
        board.hash = board.calculate_zobrist_hash();
        board
    }

    fn calculate_zobrist_hash(&self) -> u64 {
        let mut hash = 0;
        for color in 0..2 {
            for piece in 0..6 {
                let mut bitboard = self.bb_piece[color][piece];
                while bitboard > 0 {
                    let sq = pop_lsb(&mut bitboard);
                    hash ^= unsafe { ZOBRIST_PIECE[color][piece][sq] };
                }
            }
        }
        hash ^= unsafe { ZOBRIST_CASTLE[self.castling as usize] };
        if self.ep != 255 {
            hash ^= unsafe { ZOBRIST_EP[(self.ep % 8) as usize] };
        }
        if self.stm == Side::Black {
            hash ^= unsafe { ZOBRIST_STM };
        }
        hash
    }

    pub fn make_move(&mut self, m: Move) {
        let stm = self.stm;
        let from = m.from as usize;
        let to = m.to as usize;
        let capture_sq = if m.flags & FLAG_EN_PASSANT != 0 {
            if stm == Side::White { to - 8 } else { to + 8 }
        } else {
            to
        };

        let undo = Undo {
            m,
            captured: self.piece_at(self.stm.flip(), capture_sq).map_or(255, |p| p as u8),
            castling: self.castling,
            ep: self.ep,
            half_move: self.half_move,
            hash: self.hash,
        };
        self.hist.push(undo);

        let moved_piece = self.piece_at(stm, from).unwrap();

        if moved_piece == PAWN || undo.captured != 255 {
            self.half_move = 0;
        } else {
            self.half_move += 1;
        }

        if stm == Side::Black {
            self.full_move += 1;
        }

        if self.ep != 255 {
            self.hash ^= unsafe { ZOBRIST_EP[(self.ep % 8) as usize] };
        }
        self.ep = 255;

        self.bb_piece[stm as usize][moved_piece] ^= (1u64 << from) | (1u64 << to);
        self.bb_side[stm as usize] ^= (1u64 << from) | (1u64 << to);
        self.hash ^= unsafe { ZOBRIST_PIECE[stm as usize][moved_piece][from] };
        self.hash ^= unsafe { ZOBRIST_PIECE[stm as usize][moved_piece][to] };

        if undo.captured != 255 {
            let captured_piece = undo.captured as usize;
            self.bb_piece[stm.flip() as usize][captured_piece] ^= 1u64 << capture_sq;
            self.bb_side[stm.flip() as usize] ^= 1u64 << capture_sq;
            self.hash ^= unsafe { ZOBRIST_PIECE[stm.flip() as usize][captured_piece][capture_sq] };
        }

        if moved_piece == PAWN {
            if (from as i32 - to as i32).abs() == 16 {
                self.ep = ((from + to) / 2) as u8;
                self.hash ^= unsafe { ZOBRIST_EP[(self.ep % 8) as usize] };
            } else if m.promo != 255 {
                self.bb_piece[stm as usize][PAWN] ^= 1u64 << to;
                self.bb_piece[stm as usize][m.promo as usize] |= 1u64 << to;
                self.hash ^= unsafe { ZOBRIST_PIECE[stm as usize][PAWN][to] };
                self.hash ^= unsafe { ZOBRIST_PIECE[stm as usize][m.promo as usize][to] };
            }
        }

        if m.flags & FLAG_CASTLE != 0 {
            match (stm, to) {
                (Side::White, 6) => self.move_rook_for_castling(7, 5, stm),
                (Side::White, 2) => self.move_rook_for_castling(0, 3, stm),
                (Side::Black, 62) => self.move_rook_for_castling(63, 61, stm),
                (Side::Black, 58) => self.move_rook_for_castling(56, 59, stm),
                _ => {}
            }
        }

        self.hash ^= unsafe { ZOBRIST_CASTLE[self.castling as usize] };
        self.castling &= unsafe { CASTLE_MASK[from] & CASTLE_MASK[to] };
        self.hash ^= unsafe { ZOBRIST_CASTLE[self.castling as usize] };

        self.stm = self.stm.flip();
        self.hash ^= unsafe { ZOBRIST_STM };
    }

    pub fn make_null_move(&mut self) -> NullMoveState {
        let state = NullMoveState {
            ep: self.ep,
            half_move: self.half_move,
            full_move: self.full_move,
            hash: self.hash,
        };

        if self.ep != 255 {
            self.hash ^= unsafe { ZOBRIST_EP[(self.ep % 8) as usize] };
        }
        self.ep = 255;
        self.half_move += 1;

        if self.stm == Side::Black {
            self.full_move += 1;
        }

        self.stm = self.stm.flip();
        self.hash ^= unsafe { ZOBRIST_STM };
        state
    }

    pub fn unmake_null_move(&mut self, state: NullMoveState) {
        self.stm = self.stm.flip();
        self.hash ^= unsafe { ZOBRIST_STM };
        self.ep = state.ep;
        self.half_move = state.half_move;
        self.full_move = state.full_move;
        self.hash = state.hash;
    }

    fn move_rook_for_castling(&mut self, from: usize, to: usize, side: Side) {
        self.bb_piece[side as usize][ROOK] ^= (1u64 << from) | (1u64 << to);
        self.bb_side[side as usize] ^= (1u64 << from) | (1u64 << to);
        self.hash ^= unsafe { ZOBRIST_PIECE[side as usize][ROOK][from] };
        self.hash ^= unsafe { ZOBRIST_PIECE[side as usize][ROOK][to] };
    }

    pub fn unmake(&mut self) {
        let undo = self.hist.pop().unwrap();
        self.stm = self.stm.flip();
        self.castling = undo.castling;
        self.ep = undo.ep;
        self.half_move = undo.half_move;
        self.hash = undo.hash;

        if self.stm == Side::Black {
            self.full_move -= 1;
        }

        let m = undo.m;
        let from = m.from as usize;
        let to = m.to as usize;
        let stm = self.stm;

        let moved_piece = if m.promo == 255 {
            self.piece_at(stm, to).unwrap()
        } else {
            self.bb_piece[stm as usize][m.promo as usize] ^= 1u64 << to;
            self.bb_piece[stm as usize][PAWN] |= 1u64 << to;
            PAWN
        };

        self.bb_piece[stm as usize][moved_piece] ^= (1u64 << from) | (1u64 << to);
        self.bb_side[stm as usize] ^= (1u64 << from) | (1u64 << to);

        if m.flags & FLAG_CASTLE != 0 {
            match (stm, to) {
                (Side::White, 6) => {
                    self.bb_piece[stm as usize][ROOK] ^= (1u64 << 7) | (1u64 << 5);
                    self.bb_side[stm as usize] ^= (1u64 << 7) | (1u64 << 5);
                }
                (Side::White, 2) => {
                    self.bb_piece[stm as usize][ROOK] ^= 1u64 | (1u64 << 3);
                    self.bb_side[stm as usize] ^= 1u64 | (1u64 << 3);
                }
                (Side::Black, 62) => {
                    self.bb_piece[stm as usize][ROOK] ^= (1u64 << 63) | (1u64 << 61);
                    self.bb_side[stm as usize] ^= (1u64 << 63) | (1u64 << 61);
                }
                (Side::Black, 58) => {
                    self.bb_piece[stm as usize][ROOK] ^= (1u64 << 56) | (1u64 << 59);
                    self.bb_side[stm as usize] ^= (1u64 << 56) | (1u64 << 59);
                }
                _ => {}
            }
        }

        if undo.captured != 255 {
            let captured_piece = undo.captured as usize;
            let capture_sq = if m.flags & FLAG_EN_PASSANT != 0 {
                if stm == Side::White { to - 8 } else { to + 8 }
            } else {
                to
            };
            self.bb_piece[stm.flip() as usize][captured_piece] |= 1u64 << capture_sq;
            self.bb_side[stm.flip() as usize] |= 1u64 << capture_sq;
        }
    }

    pub fn in_check(&self, side: Side) -> bool {
        let king_sq = self.bb_piece[side as usize][KING].trailing_zeros() as usize;
        self.is_square_attacked(king_sq, side.flip())
    }

    pub fn is_square_attacked(&self, sq: usize, by: Side) -> bool {
        let opponent_bb = self.bb_side[by as usize];
        let own_bb = self.bb_side[by.flip() as usize];
        let occupancy = opponent_bb | own_bb;

        if self.is_attacked_by_pawn(sq, by) {
            return true;
        }

        if unsafe { KNIGHT_ATTACKS[sq] } & self.bb_piece[by as usize][KNIGHT] != 0 {
            return true;
        }
        if unsafe { KING_ATTACKS[sq] } & self.bb_piece[by as usize][KING] != 0 {
            return true;
        }
        if bishop_attacks(sq, occupancy) & (self.bb_piece[by as usize][BISHOP] | self.bb_piece[by as usize][QUEEN]) != 0 {
            return true;
        }
        if rook_attacks(sq, occupancy) & (self.bb_piece[by as usize][ROOK] | self.bb_piece[by as usize][QUEEN]) != 0 {
            return true;
        }

        false
    }

    fn is_attacked_by_pawn(&self, sq: usize, by: Side) -> bool {
        if by == Side::White {
            (sq >= 9 && sq % 8 != 0 && self.bb_piece[Side::White as usize][PAWN] & (1u64 << (sq - 9)) != 0)
                || (sq >= 7 && sq % 8 < 7 && self.bb_piece[Side::White as usize][PAWN] & (1u64 << (sq - 7)) != 0)
        } else {
            (sq < 56 && sq % 8 != 0 && self.bb_piece[Side::Black as usize][PAWN] & (1u64 << (sq + 7)) != 0)
                || (sq < 55 && sq % 8 < 7 && self.bb_piece[Side::Black as usize][PAWN] & (1u64 << (sq + 9)) != 0)
        }
    }

    pub fn gen_moves(&self, list: &mut Vec<Move>) {
        let stm = self.stm;
        let own_bb = self.bb_side[stm as usize];
        let opponent_bb = self.bb_side[stm.flip() as usize];
        let occupancy = own_bb | opponent_bb;

        self.generate_pawn_moves(list, stm, occupancy, opponent_bb);
        self.generate_en_passant_moves(list, stm);
        self.generate_knight_moves(list, stm, own_bb, opponent_bb);
        self.generate_sliding_piece_moves(list, stm, own_bb, opponent_bb, occupancy);
        self.generate_castling_moves(list, stm, occupancy);
    }

    fn generate_pawn_moves(&self, list: &mut Vec<Move>, stm: Side, occupancy: u64, opponent_bb: u64) {
        let pawns = self.bb_piece[stm as usize][PAWN];

        if stm == Side::White {
            self.generate_white_pawn_pushes(list, pawns, occupancy);
            self.generate_white_pawn_captures(list, pawns, opponent_bb);
        } else {
            self.generate_black_pawn_pushes(list, pawns, occupancy);
            self.generate_black_pawn_captures(list, pawns, opponent_bb);
        }
    }

    fn generate_white_pawn_pushes(&self, list: &mut Vec<Move>, pawns: u64, occupancy: u64) {
        let mut forward = (pawns << 8) & !occupancy;
        let mut double_push = ((forward & RANK_3) << 8) & !occupancy;

        self.add_pawn_moves(list, &mut forward, -8, 56, 0);

        while double_push > 0 {
            let to = pop_lsb(&mut double_push);
            list.push(Move { from: (to - 16) as u8, to: to as u8, promo: 255, flags: 0 });
        }
    }

    fn generate_black_pawn_pushes(&self, list: &mut Vec<Move>, pawns: u64, occupancy: u64) {
        let mut forward = (pawns >> 8) & !occupancy;
        let mut double_push = ((forward & RANK_6) >> 8) & !occupancy;

        self.add_pawn_moves(list, &mut forward, 8, 8, 0);

        while double_push > 0 {
            let to = pop_lsb(&mut double_push);
            list.push(Move { from: (to + 16) as u8, to: to as u8, promo: 255, flags: 0 });
        }
    }

    fn generate_white_pawn_captures(&self, list: &mut Vec<Move>, pawns: u64, opponent_bb: u64) {
        let mut left = ((pawns & !FILE_A) << 7) & opponent_bb;
        let mut right = ((pawns & !FILE_H) << 9) & opponent_bb;

        self.add_pawn_moves(list, &mut left, -7, 56, FLAG_CAPTURE);
        self.add_pawn_moves(list, &mut right, -9, 56, FLAG_CAPTURE);
    }

    fn generate_black_pawn_captures(&self, list: &mut Vec<Move>, pawns: u64, opponent_bb: u64) {
        let mut left = ((pawns & !FILE_A) >> 9) & opponent_bb;
        let mut right = ((pawns & !FILE_H) >> 7) & opponent_bb;

        self.add_pawn_moves(list, &mut left, 9, 8, FLAG_CAPTURE);
        self.add_pawn_moves(list, &mut right, 7, 8, FLAG_CAPTURE);
    }

    fn add_pawn_moves(&self, list: &mut Vec<Move>, bitboard: &mut u64, offset: i32, promo_rank: usize, flags: u8) {
        while *bitboard > 0 {
            let to = pop_lsb(bitboard);
            let from_sq = (to as i32 + offset) as usize;
            let is_promotion = if offset < 0 { to >= promo_rank } else { to < promo_rank };

            if is_promotion {
                for &promo in &PROMOTION_PIECES {
                    list.push(Move { from: from_sq as u8, to: to as u8, promo, flags });
                }
            } else {
                list.push(Move { from: from_sq as u8, to: to as u8, promo: 255, flags });
            }
        }
    }

    fn generate_en_passant_moves(&self, list: &mut Vec<Move>, stm: Side) {
        if self.ep == 255 {
            return;
        }

        let ep_sq = self.ep as usize;
        let pawns = self.bb_piece[stm as usize][PAWN];

        if stm == Side::White {
            self.try_add_en_passant_move(list, ep_sq, pawns, -9, 9, 0);
            self.try_add_en_passant_move(list, ep_sq, pawns, -7, 7, 7);
        } else {
            self.try_add_en_passant_move(list, ep_sq, pawns, 7, 56, 0);
            self.try_add_en_passant_move(list, ep_sq, pawns, 9, 54, 7);
        }
    }

    fn try_add_en_passant_move(&self, list: &mut Vec<Move>, ep_sq: usize, pawns: u64, offset: i32, min_sq: usize, file_constraint: usize) {
        let condition = if offset < 0 {
            ep_sq >= min_sq && ep_sq % 8 != file_constraint
        } else {
            ep_sq <= min_sq && ep_sq % 8 != file_constraint
        };

        if condition {
            let from_sq = (ep_sq as i32 + offset) as usize;
            if pawns & (1u64 << from_sq) != 0 {
                list.push(Move {
                    from: from_sq as u8,
                    to: ep_sq as u8,
                    promo: 255,
                    flags: FLAG_CAPTURE | FLAG_EN_PASSANT,
                });
            }
        }
    }

    fn generate_knight_moves(&self, list: &mut Vec<Move>, stm: Side, own_bb: u64, opponent_bb: u64) {
        let knights = self.bb_piece[stm as usize][KNIGHT];
        for from in BitboardIterator::new(knights) {
            let attacks = unsafe { KNIGHT_ATTACKS[from] } & !own_bb;
            for to in BitboardIterator::new(attacks) {
                list.push(Move {
                    from: from as u8,
                    to: to as u8,
                    promo: 255,
                    flags: if opponent_bb & (1u64 << to) > 0 { FLAG_CAPTURE } else { 0 },
                });
            }
        }
    }

    fn generate_sliding_piece_moves(&self, list: &mut Vec<Move>, stm: Side, own_bb: u64, opponent_bb: u64, occupancy: u64) {
        for piece in 2..6 {
            let mut piece_bb = self.bb_piece[stm as usize][piece];
            while piece_bb > 0 {
                let from = pop_lsb(&mut piece_bb);
                let attacks = match piece {
                    BISHOP => bishop_attacks(from, occupancy),
                    ROOK => rook_attacks(from, occupancy),
                    QUEEN => bishop_attacks(from, occupancy) | rook_attacks(from, occupancy),
                    KING => unsafe { KING_ATTACKS[from] },
                    _ => 0,
                } & !own_bb;

                for to in BitboardIterator::new(attacks) {
                    list.push(Move {
                        from: from as u8,
                        to: to as u8,
                        promo: 255,
                        flags: if opponent_bb & (1u64 << to) > 0 { FLAG_CAPTURE } else { 0 },
                    });
                }
            }
        }
    }

    fn generate_castling_moves(&self, list: &mut Vec<Move>, stm: Side, occupancy: u64) {
        let king_sq = self.bb_piece[stm as usize][KING].trailing_zeros() as usize;

        if stm == Side::White {
            if self.can_castle_kingside_white(king_sq, occupancy) {
                list.push(Move { from: 4, to: 6, promo: 255, flags: FLAG_CASTLE });
            }
            if self.can_castle_queenside_white(king_sq, occupancy) {
                list.push(Move { from: 4, to: 2, promo: 255, flags: FLAG_CASTLE });
            }
        } else {
            if self.can_castle_kingside_black(king_sq, occupancy) {
                list.push(Move { from: 60, to: 62, promo: 255, flags: FLAG_CASTLE });
            }
            if self.can_castle_queenside_black(king_sq, occupancy) {
                list.push(Move { from: 60, to: 58, promo: 255, flags: FLAG_CASTLE });
            }
        }
    }

    fn can_castle_kingside_white(&self, king_sq: usize, occupancy: u64) -> bool {
        self.castling & 1 != 0
            && king_sq == 4
            && (occupancy & ((1u64 << 5) | (1u64 << 6))) == 0
            && !self.is_square_attacked(4, Side::Black)
            && !self.is_square_attacked(5, Side::Black)
            && !self.is_square_attacked(6, Side::Black)
    }

    fn can_castle_queenside_white(&self, king_sq: usize, occupancy: u64) -> bool {
        self.castling & 2 != 0
            && king_sq == 4
            && (occupancy & ((1u64 << 1) | (1u64 << 2) | (1u64 << 3))) == 0
            && !self.is_square_attacked(4, Side::Black)
            && !self.is_square_attacked(3, Side::Black)
            && !self.is_square_attacked(2, Side::Black)
    }

    fn can_castle_kingside_black(&self, king_sq: usize, occupancy: u64) -> bool {
        self.castling & 4 != 0
            && king_sq == 60
            && (occupancy & ((1u64 << 61) | (1u64 << 62))) == 0
            && !self.is_square_attacked(60, Side::White)
            && !self.is_square_attacked(61, Side::White)
            && !self.is_square_attacked(62, Side::White)
    }

    fn can_castle_queenside_black(&self, king_sq: usize, occupancy: u64) -> bool {
        self.castling & 8 != 0
            && king_sq == 60
            && (occupancy & ((1u64 << 57) | (1u64 << 58) | (1u64 << 59))) == 0
            && !self.is_square_attacked(60, Side::White)
            && !self.is_square_attacked(59, Side::White)
            && !self.is_square_attacked(58, Side::White)
    }

    pub fn side(&self) -> usize {
        self.stm as usize
    }

    pub fn piece_bitboard(&self, side: usize, piece: usize) -> u64 {
        self.bb_piece[side][piece]
    }

    pub fn can_castle_kingside(&self, side: usize) -> bool {
        if side == 0 { self.castling & 1 != 0 } else { self.castling & 4 != 0 }
    }

    pub fn can_castle_queenside(&self, side: usize) -> bool {
        if side == 0 { self.castling & 2 != 0 } else { self.castling & 8 != 0 }
    }

    pub fn has_en_passant(&self) -> bool {
        self.ep != 255
    }

    pub fn halfmove_clock(&self) -> u8 {
        self.half_move
    }

    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::startpos()
    }

    pub fn is_insufficient_material(&self) -> bool {
        let white = Side::White as usize;
        let black = Side::Black as usize;

        if self.bb_piece[white][PAWN] != 0 || self.bb_piece[black][PAWN] != 0 {
            return false;
        }
        if self.bb_piece[white][ROOK] != 0 || self.bb_piece[black][ROOK] != 0 {
            return false;
        }
        if self.bb_piece[white][QUEEN] != 0 || self.bb_piece[black][QUEEN] != 0 {
            return false;
        }

        let white_knights = self.bb_piece[white][KNIGHT].count_ones();
        let white_bishops = self.bb_piece[white][BISHOP].count_ones();
        let black_knights = self.bb_piece[black][KNIGHT].count_ones();
        let black_bishops = self.bb_piece[black][BISHOP].count_ones();

        let white_minors = white_knights + white_bishops;
        let black_minors = black_knights + black_bishops;

        if white_minors == 0 && black_minors == 0 {
            return true;
        }

        if (white_minors == 1 && black_minors == 0) || (white_minors == 0 && black_minors == 1) {
            return true;
        }

        if white_knights == 0 && black_knights == 0 && white_bishops == 1 && black_bishops == 1 {
            let white_bishop_sq = self.bb_piece[white][BISHOP].trailing_zeros() as usize;
            let black_bishop_sq = self.bb_piece[black][BISHOP].trailing_zeros() as usize;
            let white_color = (white_bishop_sq / 8 + white_bishop_sq % 8) % 2;
            let black_color = (black_bishop_sq / 8 + black_bishop_sq % 8) % 2;

            if white_color == black_color {
                return true;
            }
        }

        false
    }

    pub fn is_fifty_move_draw(&self) -> bool {
        self.half_move >= 100
    }

    pub fn is_repetition_draw(&self) -> bool {
        let mut count = 1;
        let lookback = (self.half_move as usize).min(self.hist.len());

        for i in (2..=lookback).step_by(2) {
            if self.hist.len() >= i {
                let idx = self.hist.len() - i;
                if self.hist[idx].hash == self.hash {
                    count += 1;
                    if count >= 3 {
                        return true;
                    }
                }
            }
        }

        false
    }

    pub fn is_draw(&self) -> bool {
        self.is_insufficient_material() || self.is_fifty_move_draw() || self.is_repetition_draw()
    }
}

pub fn parse_square(s: &str) -> Option<usize> {
    if s.len() != 2 {
        return None;
    }
    let mut chars = s.chars();
    let f = chars.next()? as usize - 'a' as usize;
    let r = chars.next()? as usize - '1' as usize;
    if f > 7 || r > 7 {
        return None;
    }
    Some(r * 8 + f)
}

pub fn square_to_string(sq: usize) -> String {
    let f = (sq % 8) as u8 + b'a';
    let r = (sq / 8) as u8 + b'1';
    format!("{}{}", f as char, r as char)
}

pub fn move_to_str(m: Move) -> String {
    let mut s = format!("{}{}", square_to_string(m.from as usize), square_to_string(m.to as usize));
    if m.promo != 255 {
        s.push(match m.promo {
            1 => 'n',
            2 => 'b',
            3 => 'r',
            4 => 'q',
            _ => 'q',
        })
    }
    s
}

pub fn str_to_move(board: &Board, s: &str) -> Option<Move> {
    if s.len() < 4 {
        return None;
    }
    let from = parse_square(&s[0..2])?;
    let to = parse_square(&s[2..4])?;
    let promo = if s.len() == 5 {
        match &s[4..5] {
            "n" => 1,
            "b" => 2,
            "r" => 3,
            "q" => 4,
            _ => 4,
        }
    } else {
        255
    };

    let mut list = Vec::new();
    board.gen_moves(&mut list);
    list.into_iter().find(|&m| m.from as usize == from && m.to as usize == to && (m.promo == promo || (m.promo == 255 && promo == 255)))
}

static mut ZOBRIST_PIECE: [[[u64; 64]; 6]; 2] = [[[0; 64]; 6]; 2];
static mut ZOBRIST_CASTLE: [u64; 16] = [0; 16];
static mut ZOBRIST_EP: [u64; 8] = [0; 8];
static mut ZOBRIST_STM: u64 = 0;
static mut CASTLE_MASK: [u8; 64] = [0; 64];

pub fn init_zobrist() {
    let mut rng = XorShiftRng(0xdeadbeefcafebabe);
    unsafe {
        for c in 0..2 {
            for p in 0..6 {
                for s in 0..64 {
                    ZOBRIST_PIECE[c][p][s] = rng.next();
                }
            }
        }
        for i in 0..16 {
            ZOBRIST_CASTLE[i] = rng.next();
        }
        for i in 0..8 {
            ZOBRIST_EP[i] = rng.next();
        }
        ZOBRIST_STM = rng.next();

        for i in 0..64 {
            CASTLE_MASK[i] = 15;
        }
        CASTLE_MASK[0] &= 13;
        CASTLE_MASK[4] &= 12;
        CASTLE_MASK[7] &= 14;
        CASTLE_MASK[56] &= 7;
        CASTLE_MASK[60] &= 3;
        CASTLE_MASK[63] &= 11;
    }
}

static mut KNIGHT_ATTACKS: [u64; 64] = [0; 64];
static mut KING_ATTACKS: [u64; 64] = [0; 64];

pub fn init_tables() {
    const KNIGHT_OFFSETS: [(i32, i32); 8] = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)];

    for sq in 0..64 {
        let rank = (sq / 8) as i32;
        let file = (sq % 8) as i32;

        unsafe {
            KNIGHT_ATTACKS[sq] = 0;
            for (dr, df) in KNIGHT_OFFSETS {
                let nr = rank + dr;
                let nf = file + df;
                if (0..8).contains(&nr) && (0..8).contains(&nf) {
                    let target = (nr * 8 + nf) as u32;
                    KNIGHT_ATTACKS[sq] |= 1u64 << target;
                }
            }

            KING_ATTACKS[sq] = 0;
            for dr in -1..=1 {
                for df in -1..=1 {
                    if dr == 0 && df == 0 {
                        continue;
                    }
                    let nr = rank + dr;
                    let nf = file + df;
                    if (0..8).contains(&nr) && (0..8).contains(&nf) {
                        let target = (nr * 8 + nf) as u32;
                        KING_ATTACKS[sq] |= 1u64 << target;
                    }
                }
            }
        }
    }
}

fn bishop_attacks(sq: usize, occupancy: u64) -> u64 {
    let mut attacks = 0;
    let rank = sq / 8;
    let file = sq % 8;

    for i in 1.. {
        if rank + i > 7 || file + i > 7 {
            break;
        }
        let to = (rank + i) * 8 + (file + i);
        attacks |= 1u64 << to;
        if occupancy & (1u64 << to) > 0 {
            break;
        }
    }
    for i in 1.. {
        if rank + i > 7 || file < i {
            break;
        }
        let to = (rank + i) * 8 + (file - i);
        attacks |= 1u64 << to;
        if occupancy & (1u64 << to) > 0 {
            break;
        }
    }
    for i in 1.. {
        if rank < i || file + i > 7 {
            break;
        }
        let to = (rank - i) * 8 + (file + i);
        attacks |= 1u64 << to;
        if occupancy & (1u64 << to) > 0 {
            break;
        }
    }
    for i in 1.. {
        if rank < i || file < i {
            break;
        }
        let to = (rank - i) * 8 + (file - i);
        attacks |= 1u64 << to;
        if occupancy & (1u64 << to) > 0 {
            break;
        }
    }

    attacks
}

fn rook_attacks(sq: usize, occupancy: u64) -> u64 {
    let mut attacks = 0;
    let rank = sq / 8;
    let file = sq % 8;

    for i in 1.. {
        if rank + i > 7 {
            break;
        }
        let to = (rank + i) * 8 + file;
        attacks |= 1u64 << to;
        if occupancy & (1u64 << to) > 0 {
            break;
        }
    }
    for i in 1.. {
        if rank < i {
            break;
        }
        let to = (rank - i) * 8 + file;
        attacks |= 1u64 << to;
        if occupancy & (1u64 << to) > 0 {
            break;
        }
    }
    for i in 1.. {
        if file + i > 7 {
            break;
        }
        let to = rank * 8 + (file + i);
        attacks |= 1u64 << to;
        if occupancy & (1u64 << to) > 0 {
            break;
        }
    }
    for i in 1.. {
        if file < i {
            break;
        }
        let to = rank * 8 + (file - i);
        attacks |= 1u64 << to;
        if occupancy & (1u64 << to) > 0 {
            break;
        }
    }

    attacks
}

#[inline]
pub fn pop_lsb(b: &mut u64) -> usize {
    let s = b.trailing_zeros() as usize;
    *b &= *b - 1;
    s
}

struct XorShiftRng(u64);

impl XorShiftRng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

pub struct BitboardIterator {
    bb: u64,
}

impl BitboardIterator {
    pub fn new(bb: u64) -> Self {
        Self { bb }
    }
}

impl Iterator for BitboardIterator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bb == 0 {
            None
        } else {
            Some(pop_lsb(&mut self.bb))
        }
    }
}
