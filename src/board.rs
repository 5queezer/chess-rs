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
        if *self == Side::White {
            Side::Black
        } else {
            Side::White
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
pub struct NullState {
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
        for p in 0..6 {
            if self.bb_piece[side as usize][p] & (1u64 << sq) != 0 {
                return Some(p);
            }
        }
        None
    }

    pub fn startpos() -> Self {
        Self::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    }

    pub fn from_fen(fen: &str) -> Self {
        let mut b = Self {
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
                let side = if c.is_uppercase() {
                    Side::White
                } else {
                    Side::Black
                };
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
                b.bb_piece[side as usize][piece] |= 1u64 << sq;
                b.bb_side[side as usize] |= 1u64 << sq;
                file += 1;
            }
        }

        b.stm = if it.next().unwrap() == "w" {
            Side::White
        } else {
            Side::Black
        };

        let castling_str = it.next().unwrap();
        if castling_str.contains('K') {
            b.castling |= 1;
        }
        if castling_str.contains('Q') {
            b.castling |= 2;
        }
        if castling_str.contains('k') {
            b.castling |= 4;
        }
        if castling_str.contains('q') {
            b.castling |= 8;
        }

        let ep_str = it.next().unwrap();
        if ep_str != "-" {
            b.ep = sq_from_str(ep_str).unwrap() as u8;
        }

        b.half_move = it.next().unwrap().parse().unwrap();
        b.full_move = it.next().unwrap().parse().unwrap();

        b.hash = b.calc_hash();
        b
    }

    fn calc_hash(&self) -> u64 {
        let mut h = 0;
        for c in 0..2 {
            for p in 0..6 {
                let mut bbp = self.bb_piece[c][p];
                while bbp > 0 {
                    let sq = pop_lsb(&mut bbp);
                    h ^= unsafe { ZOBRIST_PIECE[c][p][sq] };
                }
            }
        }
        h ^= unsafe { ZOBRIST_CASTLE[self.castling as usize] };
        if self.ep != 255 {
            h ^= unsafe { ZOBRIST_EP[(self.ep % 8) as usize] };
        }
        if self.stm == Side::Black {
            h ^= unsafe { ZOBRIST_STM };
        }
        h
    }

    pub fn make_move(&mut self, m: Move) {
        let stm = self.stm;
        let from = m.from as usize;
        let to = m.to as usize;
        let capture_sq = if m.flags & FLAG_EN_PASSANT != 0 {
            if stm == Side::White {
                to - 8
            } else {
                to + 8
            }
        } else {
            to
        };
        let undo = Undo {
            m,
            captured: self
                .piece_at(self.stm.flip(), capture_sq)
                .map_or(255, |p| p as u8),
            castling: self.castling,
            ep: self.ep,
            half_move: self.half_move,
            hash: self.hash,
        };
        self.hist.push(undo);

        let moved = self.piece_at(stm, from).unwrap();

        if moved == PAWN || undo.captured != 255 {
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

        self.bb_piece[stm as usize][moved] ^= (1u64 << from) | (1u64 << to);
        self.bb_side[stm as usize] ^= (1u64 << from) | (1u64 << to);
        self.hash ^= unsafe { ZOBRIST_PIECE[stm as usize][moved][from] };
        self.hash ^= unsafe { ZOBRIST_PIECE[stm as usize][moved][to] };

        if undo.captured != 255 {
            let captured_piece = undo.captured as usize;
            self.bb_piece[stm.flip() as usize][captured_piece] ^= 1u64 << capture_sq;
            self.bb_side[stm.flip() as usize] ^= 1u64 << capture_sq;
            self.hash ^=
                unsafe { ZOBRIST_PIECE[stm.flip() as usize][captured_piece][capture_sq] };
        }

        if moved == PAWN {
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
                (Side::White, 6) => {
                    self.move_rook_for_castle(7, 5, stm);
                }
                (Side::White, 2) => {
                    self.move_rook_for_castle(0, 3, stm);
                }
                (Side::Black, 62) => {
                    self.move_rook_for_castle(63, 61, stm);
                }
                (Side::Black, 58) => {
                    self.move_rook_for_castle(56, 59, stm);
                }
                _ => {}
            }
        }

        self.hash ^= unsafe { ZOBRIST_CASTLE[self.castling as usize] };
        self.castling &= unsafe { CASTLE_MASK[from] & CASTLE_MASK[to] };
        self.hash ^= unsafe { ZOBRIST_CASTLE[self.castling as usize] };

        self.stm = self.stm.flip();
        self.hash ^= unsafe { ZOBRIST_STM };
    }

    pub fn make_null_move(&mut self) -> NullState {
        let state = NullState {
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

    pub fn unmake_null_move(&mut self, state: NullState) {
        self.stm = self.stm.flip();
        self.hash ^= unsafe { ZOBRIST_STM };
        self.ep = state.ep;
        self.half_move = state.half_move;
        self.full_move = state.full_move;
        self.hash = state.hash;
    }

    fn move_rook_for_castle(&mut self, from: usize, to: usize, side: Side) {
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

        let moved = if m.promo == 255 {
            self.piece_at(stm, to).unwrap()
        } else {
            self.bb_piece[stm as usize][m.promo as usize] ^= 1u64 << to;
            self.bb_piece[stm as usize][PAWN] |= 1u64 << to;
            PAWN
        };

        self.bb_piece[stm as usize][moved] ^= (1u64 << from) | (1u64 << to);
        self.bb_side[stm as usize] ^= (1u64 << from) | (1u64 << to);

        if m.flags & FLAG_CASTLE != 0 {
            match (stm, to) {
                (Side::White, 6) => {
                    self.bb_piece[stm as usize][ROOK] ^= (1u64 << 7) | (1u64 << 5);
                    self.bb_side[stm as usize] ^= (1u64 << 7) | (1u64 << 5);
                }
                (Side::White, 2) => {
                    self.bb_piece[stm as usize][ROOK] ^= (1u64 << 0) | (1u64 << 3);
                    self.bb_side[stm as usize] ^= (1u64 << 0) | (1u64 << 3);
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
                if stm == Side::White {
                    to - 8
                } else {
                    to + 8
                }
            } else {
                to
            };
            self.bb_piece[stm.flip() as usize][captured_piece] |= 1u64 << capture_sq;
            self.bb_side[stm.flip() as usize] |= 1u64 << capture_sq;
        }
    }

    pub fn in_check(&self, side: Side) -> bool {
        let ksq = self.bb_piece[side as usize][KING].trailing_zeros() as usize;
        self.is_attacked(ksq, side.flip())
    }

    pub fn is_attacked(&self, sq: usize, by: Side) -> bool {
        let bb_opp = self.bb_side[by as usize];
        let bb_own = self.bb_side[by.flip() as usize];
        let bb_occ = bb_opp | bb_own;

        if by == Side::White {
            if sq >= 9
                && (sq % 8 > 0)
                && self.bb_piece[Side::White as usize][PAWN] & (1u64 << (sq - 9)) != 0
            {
                return true;
            }
            if sq >= 7
                && (sq % 8 < 7)
                && self.bb_piece[Side::White as usize][PAWN] & (1u64 << (sq - 7)) != 0
            {
                return true;
            }
        } else {
            if sq < 56
                && (sq % 8 > 0)
                && self.bb_piece[Side::Black as usize][PAWN] & (1u64 << (sq + 7)) != 0
            {
                return true;
            }
            if sq < 55
                && (sq % 8 < 7)
                && self.bb_piece[Side::Black as usize][PAWN] & (1u64 << (sq + 9)) != 0
            {
                return true;
            }
        }

        if unsafe { NATT[sq] } & self.bb_piece[by as usize][KNIGHT] != 0 { return true; }
        if unsafe { KATT[sq] } & self.bb_piece[by as usize][KING] != 0 { return true; }
        if bishop_attacks(sq, bb_occ) & (self.bb_piece[by as usize][BISHOP] | self.bb_piece[by as usize][QUEEN]) != 0 { return true; }
        if rook_attacks(sq, bb_occ) & (self.bb_piece[by as usize][ROOK] | self.bb_piece[by as usize][QUEEN]) != 0 { return true; }

        false
    }

    pub fn gen_moves(&self, list: &mut Vec<Move>) {
        let stm = self.stm;
        let bb_own = self.bb_side[stm as usize];
        let bb_opp = self.bb_side[stm.flip() as usize];
        let bb_occ = bb_own | bb_opp;

        let pawns = self.bb_piece[stm as usize][PAWN];
        if stm == Side::White {
            let mut fwd = (pawns << 8) & !bb_occ;
            let mut dbl = ((fwd & RANK_3) << 8) & !bb_occ;
            while fwd > 0 {
                let to = pop_lsb(&mut fwd);
                let from_sq = to - 8;
                if to >= 56 {
                    for &promo in &PROMOTION_PIECES {
                        list.push(Move { from: from_sq as u8, to: to as u8, promo, flags: 0 });
                    }
                } else {
                    list.push(Move { from: from_sq as u8, to: to as u8, promo: 255, flags: 0 });
                }
            }
            while dbl > 0 {
                let to = pop_lsb(&mut dbl);
                list.push(Move { from: (to - 16) as u8, to: to as u8, promo: 255, flags: 0 });
            }
            let mut l = ((pawns & !FILE_A) << 7) & bb_opp;
            let mut r = ((pawns & !FILE_H) << 9) & bb_opp;
            while l > 0 {
                let to = pop_lsb(&mut l);
                let from_sq = to - 7;
                if to >= 56 {
                    for &promo in &PROMOTION_PIECES {
                        list.push(Move { from: from_sq as u8, to: to as u8, promo, flags: FLAG_CAPTURE });
                    }
                } else {
                    list.push(Move { from: from_sq as u8, to: to as u8, promo: 255, flags: FLAG_CAPTURE });
                }
            }
            while r > 0 {
                let to = pop_lsb(&mut r);
                let from_sq = to - 9;
                if to >= 56 {
                    for &promo in &PROMOTION_PIECES {
                        list.push(Move { from: from_sq as u8, to: to as u8, promo, flags: FLAG_CAPTURE });
                    }
                } else {
                    list.push(Move { from: from_sq as u8, to: to as u8, promo: 255, flags: FLAG_CAPTURE });
                }
            }
        } else {
            let mut fwd = (pawns >> 8) & !bb_occ;
            let mut dbl = ((fwd & RANK_6) >> 8) & !bb_occ;
            while fwd > 0 {
                let to = pop_lsb(&mut fwd);
                let from_sq = to + 8;
                if to < 8 {
                    for &promo in &PROMOTION_PIECES {
                        list.push(Move { from: from_sq as u8, to: to as u8, promo, flags: 0 });
                    }
                } else {
                    list.push(Move { from: from_sq as u8, to: to as u8, promo: 255, flags: 0 });
                }
            }
            while dbl > 0 {
                let to = pop_lsb(&mut dbl);
                list.push(Move { from: (to + 16) as u8, to: to as u8, promo: 255, flags: 0 });
            }
            let mut l = ((pawns & !FILE_A) >> 9) & bb_opp;
            let mut r = ((pawns & !FILE_H) >> 7) & bb_opp;
            while l > 0 {
                let to = pop_lsb(&mut l);
                let from_sq = to + 9;
                if to < 8 {
                    for &promo in &PROMOTION_PIECES {
                        list.push(Move { from: from_sq as u8, to: to as u8, promo, flags: FLAG_CAPTURE });
                    }
                } else {
                    list.push(Move { from: from_sq as u8, to: to as u8, promo: 255, flags: FLAG_CAPTURE });
                }
            }
            while r > 0 {
                let to = pop_lsb(&mut r);
                let from_sq = to + 7;
                if to < 8 {
                    for &promo in &PROMOTION_PIECES {
                        list.push(Move { from: from_sq as u8, to: to as u8, promo, flags: FLAG_CAPTURE });
                    }
                } else {
                    list.push(Move { from: from_sq as u8, to: to as u8, promo: 255, flags: FLAG_CAPTURE });
                }
            }
        }

        if self.ep != 255 {
            let ep_sq = self.ep as usize;
            if stm == Side::White {
                if ep_sq >= 9 && ep_sq % 8 != 0 {
                    let from_sq = ep_sq - 9;
                    if pawns & (1u64 << from_sq) != 0 {
                        list.push(Move { from: from_sq as u8, to: ep_sq as u8, promo: 255, flags: FLAG_CAPTURE | FLAG_EN_PASSANT });
                    }
                }
                if ep_sq >= 7 && ep_sq % 8 != 7 {
                    let from_sq = ep_sq - 7;
                    if pawns & (1u64 << from_sq) != 0 {
                        list.push(Move { from: from_sq as u8, to: ep_sq as u8, promo: 255, flags: FLAG_CAPTURE | FLAG_EN_PASSANT });
                    }
                }
            } else {
                if ep_sq <= 56 && ep_sq % 8 != 0 {
                    let from_sq = ep_sq + 7;
                    if pawns & (1u64 << from_sq) != 0 {
                        list.push(Move { from: from_sq as u8, to: ep_sq as u8, promo: 255, flags: FLAG_CAPTURE | FLAG_EN_PASSANT });
                    }
                }
                if ep_sq <= 54 && ep_sq % 8 != 7 {
                    let from_sq = ep_sq + 9;
                    if pawns & (1u64 << from_sq) != 0 {
                        list.push(Move { from: from_sq as u8, to: ep_sq as u8, promo: 255, flags: FLAG_CAPTURE | FLAG_EN_PASSANT });
                    }
                }
            }
        }

        let knights = self.bb_piece[stm as usize][KNIGHT];
        for from in Bitscan::new(knights) {
            let mut attacks = unsafe { NATT[from] };
            attacks &= !bb_own;
            for to in Bitscan::new(attacks) {
                list.push(Move {
                    from: from as u8,
                    to: to as u8,
                    promo: 255,
                    flags: if bb_opp & (1u64 << to) > 0 { FLAG_CAPTURE } else { 0 },
                });
            }
        }

        for p in 2..6 {
            let mut bbp = self.bb_piece[stm as usize][p];
            while bbp > 0 {
                let from = pop_lsb(&mut bbp);
                let mut attacks = match p {
                    BISHOP => bishop_attacks(from, bb_occ),
                    ROOK => rook_attacks(from, bb_occ),
                    QUEEN => bishop_attacks(from, bb_occ) | rook_attacks(from, bb_occ),
                    KING => unsafe { KATT[from] },
                    _ => 0,
                };
                attacks &= !bb_own;
                while attacks > 0 {
                    let to = pop_lsb(&mut attacks);
                    list.push(Move {
                        from: from as u8,
                        to: to as u8,
                        promo: 255,
                        flags: if bb_opp & (1u64 << to) > 0 { FLAG_CAPTURE } else { 0 },
                    });
                }
            }
        }

        let king_sq = self.bb_piece[stm as usize][KING].trailing_zeros() as usize;
        if stm == Side::White {
            if self.castling & 1 != 0
                && king_sq == 4
                && (bb_occ & ((1u64 << 5) | (1u64 << 6))) == 0
                && !self.is_attacked(4, Side::Black)
                && !self.is_attacked(5, Side::Black)
                && !self.is_attacked(6, Side::Black)
            {
                list.push(Move { from: 4, to: 6, promo: 255, flags: FLAG_CASTLE });
            }
            if self.castling & 2 != 0
                && king_sq == 4
                && (bb_occ & ((1u64 << 1) | (1u64 << 2) | (1u64 << 3))) == 0
                && !self.is_attacked(4, Side::Black)
                && !self.is_attacked(3, Side::Black)
                && !self.is_attacked(2, Side::Black)
            {
                list.push(Move { from: 4, to: 2, promo: 255, flags: FLAG_CASTLE });
            }
        } else {
            if self.castling & 4 != 0
                && king_sq == 60
                && (bb_occ & ((1u64 << 61) | (1u64 << 62))) == 0
                && !self.is_attacked(60, Side::White)
                && !self.is_attacked(61, Side::White)
                && !self.is_attacked(62, Side::White)
            {
                list.push(Move { from: 60, to: 62, promo: 255, flags: FLAG_CASTLE });
            }
            if self.castling & 8 != 0
                && king_sq == 60
                && (bb_occ & ((1u64 << 57) | (1u64 << 58) | (1u64 << 59))) == 0
                && !self.is_attacked(60, Side::White)
                && !self.is_attacked(59, Side::White)
                && !self.is_attacked(58, Side::White)
            {
                list.push(Move { from: 60, to: 58, promo: 255, flags: FLAG_CASTLE });
            }
        }
    }
}

pub fn sq_from_str(s: &str) -> Option<usize> {
    if s.len() != 2 { return None; }
    let mut chars = s.chars();
    let f = chars.next()? as usize - 'a' as usize;
    let r = chars.next()? as usize - '1' as usize;
    if f > 7 || r > 7 { return None; }
    Some(r * 8 + f)
}

pub fn sq_to_str(sq: usize) -> String {
    let f = (sq % 8) as u8 + b'a';
    let r = (sq / 8) as u8 + b'1';
    format!("{}{}", f as char, r as char)
}

pub fn move_to_str(m: Move) -> String {
    let mut s = format!("{}{}", sq_to_str(m.from as usize), sq_to_str(m.to as usize));
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

pub fn str_to_move(b: &Board, s: &str) -> Option<Move> {
    if s.len() < 4 {
        return None;
    }
    let from = sq_from_str(&s[0..2])?;
    let to = sq_from_str(&s[2..4])?;
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
    b.gen_moves(&mut list);
    for m in list {
        if m.from as usize == from
            && m.to as usize == to
            && (m.promo == promo || (m.promo == 255 && promo == 255))
        {
            return Some(m);
        }
    }
    None
}

static mut ZOBRIST_PIECE: [[[u64; 64]; 6]; 2] = [[[0; 64]; 6]; 2];
static mut ZOBRIST_CASTLE: [u64; 16] = [0; 16];
static mut ZOBRIST_EP: [u64; 8] = [0; 8];
static mut ZOBRIST_STM: u64 = 0;
static mut CASTLE_MASK: [u8; 64] = [0; 64];

pub fn init_zobrist() {
    let mut rng = XorShift(0xdeadbeefcafebabe);
    unsafe {
        for c in 0..2 { for p in 0..6 { for s in 0..64 { ZOBRIST_PIECE[c][p][s] = rng.rand(); } } }
        for i in 0..16 { ZOBRIST_CASTLE[i] = rng.rand(); }
        for i in 0..8 { ZOBRIST_EP[i] = rng.rand(); }
        ZOBRIST_STM = rng.rand();

        for i in 0..64 { CASTLE_MASK[i] = 15; }
        CASTLE_MASK[0] &= 13; CASTLE_MASK[4] &= 12; CASTLE_MASK[7] &= 14;
        CASTLE_MASK[56] &= 7; CASTLE_MASK[60] &= 3; CASTLE_MASK[63] &= 11;
    }
}

static mut NATT: [u64; 64] = [0; 64];
static mut KATT: [u64; 64] = [0; 64];

pub fn init_tables() {
    const KNIGHT_DIRS: [(i32, i32); 8] = [
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    ];

    for sq in 0..64 {
        let rank = (sq / 8) as i32;
        let file = (sq % 8) as i32;

        unsafe {
            NATT[sq] = 0;
            for (dr, df) in KNIGHT_DIRS {
                let nr = rank + dr;
                let nf = file + df;
                if (0..8).contains(&nr) && (0..8).contains(&nf) {
                    let target = (nr * 8 + nf) as u32;
                    NATT[sq] |= 1u64 << target;
                }
            }

            KATT[sq] = 0;
            for dr in -1..=1 {
                for df in -1..=1 {
                    if dr == 0 && df == 0 {
                        continue;
                    }
                    let nr = rank + dr;
                    let nf = file + df;
                    if (0..8).contains(&nr) && (0..8).contains(&nf) {
                        let target = (nr * 8 + nf) as u32;
                        KATT[sq] |= 1u64 << target;
                    }
                }
            }
        }
    }
}

fn bishop_attacks(sq: usize, occ: u64) -> u64 {
    let mut attacks = 0;
    let r = sq / 8;
    let f = sq % 8;

    for i in 1.. {
        if r + i > 7 || f + i > 7 { break; }
        let to = (r + i) * 8 + (f + i);
        attacks |= 1u64 << to;
        if occ & (1u64 << to) > 0 { break; }
    }
    for i in 1.. {
        if r + i > 7 || f < i { break; }
        let to = (r + i) * 8 + (f - i);
        attacks |= 1u64 << to;
        if occ & (1u64 << to) > 0 { break; }
    }
    for i in 1.. {
        if r < i || f + i > 7 { break; }
        let to = (r - i) * 8 + (f + i);
        attacks |= 1u64 << to;
        if occ & (1u64 << to) > 0 { break; }
    }
    for i in 1.. {
        if r < i || f < i { break; }
        let to = (r - i) * 8 + (f - i);
        attacks |= 1u64 << to;
        if occ & (1u64 << to) > 0 { break; }
    }
    attacks
}

fn rook_attacks(sq: usize, occ: u64) -> u64 {
    let mut attacks = 0;
    let r = sq / 8;
    let f = sq % 8;

    for i in 1.. {
        if r + i > 7 { break; }
        let to = (r + i) * 8 + f;
        attacks |= 1u64 << to;
        if occ & (1u64 << to) > 0 { break; }
    }
    for i in 1.. {
        if r < i { break; }
        let to = (r - i) * 8 + f;
        attacks |= 1u64 << to;
        if occ & (1u64 << to) > 0 { break; }
    }
    for i in 1.. {
        if f + i > 7 { break; }
        let to = r * 8 + (f + i);
        attacks |= 1u64 << to;
        if occ & (1u64 << to) > 0 { break; }
    }
    for i in 1.. {
        if f < i { break; }
        let to = r * 8 + (f - i);
        attacks |= 1u64 << to;
        if occ & (1u64 << to) > 0 { break; }
    }
    attacks
}

#[inline]
pub fn pop_lsb(b: &mut u64) -> usize {
    let s = b.trailing_zeros() as usize;
    *b &= *b - 1;
    s
}

struct XorShift(u64);
impl XorShift { fn rand(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 } }

pub struct Bitscan {
    bb: u64,
}

impl Bitscan {
    pub fn new(bb: u64) -> Self {
        Self { bb }
    }
}

impl Iterator for Bitscan {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        if self.bb == 0 {
            None
        } else {
            Some(pop_lsb(&mut self.bb))
        }
    }
}
