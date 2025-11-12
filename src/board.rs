use std::collections::HashMap;

#[inline]
pub fn bb(sq: usize) -> u64 {
    1u64 << sq
}
#[inline]
pub fn lsb(x: u64) -> usize {
    x.trailing_zeros() as usize
}
#[inline]
pub fn pop_lsb(x: &mut u64) -> usize {
    let s = lsb(*x);
    *x &= *x - 1;
    s
}

pub const FILE_A: u64 = 0x0101010101010101;
pub const FILE_H: u64 = 0x8080808080808080;
pub const RANK_3: u64 = 0x0000000000FF0000;
pub const RANK_6: u64 = 0x0000FF0000000000;

pub fn sq_from_str(s: &str) -> Option<usize> {
    if s.len() != 2 {
        return None;
    }
    let b = s.as_bytes();
    let f = (b[0] as char).to_ascii_lowercase() as u8;
    let r = (b[1] as char) as u8;
    if !(b"abcdefgh".contains(&f) && b"12345678".contains(&r)) {
        return None;
    }
    Some(((r - b'1') as usize) * 8 + ((f - b'a') as usize))
}

pub fn sq_to_str(sq: usize) -> String {
    format!(
        "{}{}",
        (b'a' + (sq & 7) as u8) as char,
        (b'1' + (sq / 8) as u8) as char
    )
}

pub static mut KNIGHT: [u64; 64] = [0; 64];
pub static mut KING: [u64; 64] = [0; 64];
pub static mut PAWN_ATK: [[u64; 64]; 2] = [[0; 64], [0; 64]];

pub fn init_tables() {
    unsafe {
        for s in 0..64 {
            let f = s & 7;
            let r = s >> 3;
            let mut m = 0u64;
            for (df, dr) in [
                (1, 2),
                (2, 1),
                (2, -1),
                (1, -2),
                (-1, -2),
                (-2, -1),
                (-2, 1),
                (-1, 2),
            ] {
                let nf = f as i32 + df;
                let nr = r as i32 + dr;
                if nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
                    m |= bb((nr * 8 + nf) as usize)
                }
            }
            KNIGHT[s] = m;

            let mut k = 0u64;
            for df in -1..=1 {
                for dr in -1..=1 {
                    if df == 0 && dr == 0 {
                        continue;
                    }
                    let nf = f as i32 + df;
                    let nr = r as i32 + dr;
                    if nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
                        k |= bb((nr * 8 + nf) as usize)
                    }
                }
            }
            KING[s] = k;

            let mut w = 0u64;
            if f > 0 && r < 7 {
                w |= bb(s + 7)
            }
            if f < 7 && r < 7 {
                w |= bb(s + 9)
            }
            PAWN_ATK[0][s] = w;
            let mut b = 0u64;
            if f > 0 && r > 0 {
                b |= bb(s - 9)
            }
            if f < 7 && r > 0 {
                b |= bb(s - 7)
            }
            PAWN_ATK[1][s] = b;
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    White = 0,
    Black = 1,
}
impl Side {
    pub fn idx(self) -> usize {
        if self == Side::White {
            0
        } else {
            1
        }
    }
    pub fn flip(self) -> Side {
        if self == Side::White {
            Side::Black
        } else {
            Side::White
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Piece {
    P,
    N,
    B,
    R,
    Q,
    K,
}

#[derive(Clone, Copy, Debug)]
pub struct Move {
    pub from: u8,
    pub to: u8,
    pub promo: u8,
    pub flags: u8,
    pub captured: u8,
    pub moved: u8,
}
impl Move {
    pub fn quiet(from: usize, to: usize, moved: u8) -> Self {
        Self {
            from: from as u8,
            to: to as u8,
            promo: 255,
            flags: 0,
            captured: 6,
            moved,
        }
    }
    pub fn capture(from: usize, to: usize, captured: u8, moved: u8) -> Self {
        Self {
            from: from as u8,
            to: to as u8,
            promo: 255,
            flags: 1,
            captured,
            moved,
        }
    }
    pub fn promo(from: usize, to: usize, p: u8, cap: bool, captured: u8, moved: u8) -> Self {
        Self {
            from: from as u8,
            to: to as u8,
            promo: p,
            flags: if cap {
                1
            } else {
                0
            },
            captured,
            moved,
        }
    }
}

#[derive(Clone, Debug)]
pub struct State {
    pub castling: u8,
    pub ep: u8,
    pub halfmove: u16,
    pub hash: u64,
}

#[derive(Clone, Debug)]
pub struct Board {
    pub bb_side: [u64; 2],
    pub bb_piece: [[u64; 6]; 2],
    pub occ: u64,
    pub stm: Side,
    pub castling: u8,
    pub ep: u8,
    pub halfmove: u16,
    pub fullmove: u16,
    pub hash: u64,
    pub hist: Vec<State>,
    pub move_hist: Vec<Move>,
}

pub static mut ZP: [[[u64; 64]; 6]; 2] = [[[0; 64]; 6]; 2];
pub static mut ZCASTLE: [u64; 16] = [0; 16];
pub static mut ZEP: [u64; 65] = [0; 65];
pub static mut ZSTM: u64 = 0;

pub fn rng64(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    *seed
}

pub fn init_zobrist() {
    let mut s = 0x9E3779B97F4A7C15u64;
    unsafe {
        for c in 0..2 {
            for p in 0..6 {
                for sq in 0..64 {
                    ZP[c][p][sq] = rng64(&mut s);
                }
            }
        }
        for i in 0..16 {
            ZCASTLE[i] = rng64(&mut s);
        }
        for i in 0..65 {
            ZEP[i] = rng64(&mut s);
        }
        ZSTM = rng64(&mut s);
    }
}

impl Board {
    pub fn empty() -> Self {
        Self {
            bb_side: [0; 2],
            bb_piece: [[0; 6]; 2],
            occ: 0,
            stm: Side::White,
            castling: 0,
            ep: 64,
            halfmove: 0,
            fullmove: 1,
            hash: 0,
            hist: Vec::new(),
            move_hist: Vec::new(),
        }
    }
    pub fn compute_occ(&mut self) {
        self.bb_side[0] = 0;
        self.bb_side[1] = 0;
        for s in 0..2 {
            for p in 0..6 {
                self.bb_side[s] |= self.bb_piece[s][p];
            }
        }
        self.occ = self.bb_side[0] | self.bb_side[1];
    }
    pub fn update_hash(&mut self) {
        unsafe {
            self.hash = 0;
            for c in 0..2 {
                for p in 0..6 {
                    let mut b = self.bb_piece[c][p];
                    while b != 0 {
                        let s = pop_lsb(&mut b);
                        self.hash ^= ZP[c][p][s];
                    }
                }
            }
            self.hash ^= ZCASTLE[self.castling as usize];
            self.hash ^= ZEP[self.ep as usize];
            if self.stm == Side::Black {
                self.hash ^= ZSTM;
            }
        }
    }

    pub fn from_fen(fen: &str) -> Self {
        let mut b = Board::empty();
        let parts: Vec<&str> = fen.split_whitespace().collect();
        let mut sq = 56;
        for ch in parts[0].chars() {
            match ch {
                '/' => {
                    sq -= 16;
                }
                '1'..='8' => {
                    sq += ch.to_digit(10).unwrap() as usize;
                }
                _ => {
                    let (side, piece) = match ch {
                        'P' => (0, 0),
                        'N' => (0, 1),
                        'B' => (0, 2),
                        'R' => (0, 3),
                        'Q' => (0, 4),
                        'K' => (0, 5),
                        'p' => (1, 0),
                        'n' => (1, 1),
                        'b' => (1, 2),
                        'r' => (1, 3),
                        'q' => (1, 4),
                        'k' => (1, 5),
                        _ => panic!("bad FEN"),
                    };
                    b.bb_piece[side][piece] |= bb(sq);
                    sq += 1;
                }
            }
        }
        b.stm = if parts[1] == "w" {
            Side::White
        } else {
            Side::Black
        };
        let mut c = 0u8;
        if parts[2].contains('K') {
            c |= 1
        }
        if parts[2].contains('Q') {
            c |= 2
        }
        if parts[2].contains('k') {
            c |= 4
        }
        if parts[2].contains('q') {
            c |= 8
        }
        b.castling = c;
        b.ep = if parts[3] == "-" {
            64
        } else {
            sq_from_str(parts[3]).unwrap() as u8
        };
        if parts.len() > 4 {
            b.halfmove = parts[4].parse().unwrap_or(0);
        }
        if parts.len() > 5 {
            b.fullmove = parts[5].parse().unwrap_or(1);
        }
        b.compute_occ();
        b.update_hash();
        b
    }

    pub fn startpos() -> Self {
        Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    }

    pub fn in_check(&self, side: Side) -> bool {
        let king_sq = lsb(self.bb_piece[side.idx()][Piece::K as usize]);
        self.square_attacked(king_sq, side.flip())
    }

    pub fn square_attacked(&self, sq: usize, by: Side) -> bool {
        unsafe {
            if (PAWN_ATK[by.idx()][sq] & self.bb_piece[by.idx()][Piece::P as usize]) != 0 {
                return true;
            }
            if (KNIGHT[sq] & self.bb_piece[by.idx()][Piece::N as usize]) != 0 {
                return true;
            }
            if (KING[sq] & self.bb_piece[by.idx()][Piece::K as usize]) != 0 {
                return true;
            }
        }
        if (self.ray_attacks(sq, &[-9, -7, 7, 9])
            & (self.bb_piece[by.idx()][Piece::B as usize]
                | self.bb_piece[by.idx()][Piece::Q as usize]))
            != 0
        {
            return true;
        }
        if (self.ray_attacks(sq, &[-8, -1, 1, 8])
            & (self.bb_piece[by.idx()][Piece::R as usize]
                | self.bb_piece[by.idx()][Piece::Q as usize]))
            != 0
        {
            return true;
        }
        false
    }

    pub fn ray_attacks(&self, from: usize, deltas: &[i32]) -> u64 {
        let mut atk = 0u64;
        let f0 = (from & 7) as i32;
        let r0 = (from / 8) as i32;
        for &d in deltas {
            let mut s = from as i32 + d;
            loop {
                if s < 0 || s >= 64 {
                    break;
                }
                let sf = (s & 7) as i32;
                let sr = (s / 8) as i32;
                if (d == 1 || d == -1) && sr != r0 {
                    break;
                }
                if (d == 9 || d == -9 || d == 7 || d == -7) && (sf - f0).abs() != ((sr - r0).abs())
                {
                    break;
                }
                atk |= bb(s as usize);
                if (self.occ & bb(s as usize)) != 0 {
                    break;
                }
                s += d;
            }
        }
        atk
    }

    pub fn gen_moves(&self, list: &mut Vec<Move>) {
        let us = self.stm.idx();
        let them = self.stm.flip().idx();
        let occ_us = self.bb_side[us];
        let occ_them = self.bb_side[them];
        let empty = !self.occ;
        unsafe {
            let pawns = self.bb_piece[us][Piece::P as usize];
            if self.stm == Side::White {
                let single = (pawns << 8) & empty;
                let double = ((single & RANK_3) << 8) & empty;
                let mut m = single;
                while m != 0 {
                    let to = pop_lsb(&mut m);
                    let from = to - 8;
                    if to >= 56 {
                        for &p in &[Piece::N, Piece::B, Piece::R, Piece::Q] {
                            list.push(Move::promo(from, to, p as u8, false, 6, Piece::P as u8));
                        }
                    } else {
                        list.push(Move::quiet(from, to, Piece::P as u8));
                    }
                }
                let mut d = double;
                while d != 0 {
                    let to = pop_lsb(&mut d);
                    list.push(Move {
                        from: (to - 16) as u8,
                        to: to as u8,
                        promo: 255,
                        flags: 8,
                        captured: 6,
                        moved: Piece::P as u8,
                    });
                }
                let left = ((pawns & !FILE_A) << 7) & occ_them;
                let right = ((pawns & !FILE_H) << 9) & occ_them;
                let mut c = left;
                while c != 0 {
                    let to = pop_lsb(&mut c);
                    let from = to - 7;
                    let captured = self.piece_at(to, self.stm.flip()).unwrap();
                    if to >= 56 {
                        for &p in &[Piece::N, Piece::B, Piece::R, Piece::Q] {
                            list.push(Move::promo(
                                from,
                                to,
                                p as u8,
                                true,
                                captured as u8,
                                Piece::P as u8,
                            ));
                        }
                    } else {
                        list.push(Move::capture(from, to, captured as u8, Piece::P as u8));
                    }
                }
                let mut c2 = right;
                while c2 != 0 {
                    let to = pop_lsb(&mut c2);
                    let from = to - 9;
                    let captured = self.piece_at(to, self.stm.flip()).unwrap();
                    if to >= 56 {
                        for &p in &[Piece::N, Piece::B, Piece::R, Piece::Q] {
                            list.push(Move::promo(
                                from,
                                to,
                                p as u8,
                                true,
                                captured as u8,
                                Piece::P as u8,
                            ));
                        }
                    } else {
                        list.push(Move::capture(from, to, captured as u8, Piece::P as u8));
                    }
                }
                if self.ep != 64 {
                    let epb = bb(self.ep as usize);
                    let left_ep = ((pawns & !FILE_A) << 7) & epb;
                    if left_ep != 0 {
                        let to = lsb(left_ep);
                        list.push(Move {
                            from: (to - 7) as u8,
                            to: to as u8,
                            promo: 255,
                            flags: 2,
                            captured: Piece::P as u8,
                            moved: Piece::P as u8,
                        });
                    }
                    let right_ep = ((pawns & !FILE_H) << 9) & epb;
                    if right_ep != 0 {
                        let to = lsb(right_ep);
                        list.push(Move {
                            from: (to - 9) as u8,
                            to: to as u8,
                            promo: 255,
                            flags: 2,
                            captured: Piece::P as u8,
                            moved: Piece::P as u8,
                        });
                    }
                }
            } else {
                let single = (pawns >> 8) & empty;
                let double = ((single & RANK_6) >> 8) & empty;
                let mut m = single;
                while m != 0 {
                    let to = pop_lsb(&mut m);
                    let from = to + 8;
                    if to < 8 {
                        for &p in &[Piece::N, Piece::B, Piece::R, Piece::Q] {
                            list.push(Move::promo(from, to, p as u8, false, 6, Piece::P as u8));
                        }
                    } else {
                        list.push(Move::quiet(from, to, Piece::P as u8));
                    }
                }
                let mut d = double;
                while d != 0 {
                    let to = pop_lsb(&mut d);
                    list.push(Move {
                        from: (to + 16) as u8,
                        to: to as u8,
                        promo: 255,
                        flags: 8,
                        captured: 6,
                        moved: Piece::P as u8,
                    });
                }
                let left = ((pawns & !FILE_H) >> 7) & occ_them;
                let right = ((pawns & !FILE_A) >> 9) & occ_them;
                let mut c = left;
                while c != 0 {
                    let to = pop_lsb(&mut c);
                    let from = to + 7;
                    let captured = self.piece_at(to, self.stm.flip()).unwrap();
                    if to < 8 {
                        for &p in &[Piece::N, Piece::B, Piece::R, Piece::Q] {
                            list.push(Move::promo(
                                from,
                                to,
                                p as u8,
                                true,
                                captured as u8,
                                Piece::P as u8,
                            ));
                        }
                    } else {
                        list.push(Move::capture(from, to, captured as u8, Piece::P as u8));
                    }
                }
                let mut c2 = right;
                while c2 != 0 {
                    let to = pop_lsb(&mut c2);
                    let from = to + 9;
                    let captured = self.piece_at(to, self.stm.flip()).unwrap();
                    if to < 8 {
                        for &p in &[Piece::N, Piece::B, Piece::R, Piece::Q] {
                            list.push(Move::promo(
                                from,
                                to,
                                p as u8,
                                true,
                                captured as u8,
                                Piece::P as u8,
                            ));
                        }
                    } else {
                        list.push(Move::capture(from, to, captured as u8, Piece::P as u8));
                    }
                }
                if self.ep != 64 {
                    let epb = bb(self.ep as usize);
                    let left_ep = ((pawns & !FILE_H) >> 7) & epb;
                    if left_ep != 0 {
                        let to = lsb(left_ep);
                        list.push(Move {
                            from: (to + 7) as u8,
                            to: to as u8,
                            promo: 255,
                            flags: 2,
                            captured: Piece::P as u8,
                            moved: Piece::P as u8,
                        });
                    }
                    let right_ep = ((pawns & !FILE_A) >> 9) & epb;
                    if right_ep != 0 {
                        let to = lsb(right_ep);
                        list.push(Move {
                            from: (to + 9) as u8,
                            to: to as u8,
                            promo: 255,
                            flags: 2,
                            captured: Piece::P as u8,
                            moved: Piece::P as u8,
                        });
                    }
                }
            }
            let mut n = self.bb_piece[us][Piece::N as usize];
            while n != 0 {
                let s = pop_lsb(&mut n);
                let mut m = KNIGHT[s] & !occ_us;
                while m != 0 {
                    let t = pop_lsb(&mut m);
                    if (occ_them & bb(t)) != 0 {
                        let captured = self.piece_at(t, self.stm.flip()).unwrap();
                        list.push(Move::capture(s, t, captured as u8, Piece::N as u8));
                    } else {
                        list.push(Move::quiet(s, t, Piece::N as u8));
                    }
                }
            }
            let ksq = lsb(self.bb_piece[us][Piece::K as usize]);
            let mut km = KING[ksq] & !occ_us;
            while km != 0 {
                let t = pop_lsb(&mut km);
                if (occ_them & bb(t)) != 0 {
                    let captured = self.piece_at(t, self.stm.flip()).unwrap();
                    list.push(Move::capture(ksq, t, captured as u8, Piece::K as u8));
                } else {
                    list.push(Move::quiet(ksq, t, Piece::K as u8));
                }
            }
            if self.stm == Side::White {
                if (self.castling & 1) != 0
                    && (self.occ & (bb(5) | bb(6))) == 0
                    && !self.square_attacked(4, Side::Black)
                    && !self.square_attacked(5, Side::Black)
                    && !self.square_attacked(6, Side::Black)
                {
                    list.push(Move {
                        from: 4,
                        to: 6,
                        promo: 255,
                        flags: 4,
                        captured: 6,
                        moved: Piece::K as u8,
                    });
                }
                if (self.castling & 2) != 0
                    && (self.occ & (bb(3) | bb(2) | bb(1))) == 0
                    && !self.square_attacked(4, Side::Black)
                    && !self.square_attacked(3, Side::Black)
                    && !self.square_attacked(2, Side::Black)
                {
                    list.push(Move {
                        from: 4,
                        to: 2,
                        promo: 255,
                        flags: 4,
                        captured: 6,
                        moved: Piece::K as u8,
                    });
                }
            } else {
                if (self.castling & 4) != 0
                    && (self.occ & (bb(61) | bb(62))) == 0
                    && !self.square_attacked(60, Side::White)
                    && !self.square_attacked(61, Side::White)
                    && !self.square_attacked(62, Side::White)
                {
                    list.push(Move {
                        from: 60,
                        to: 62,
                        promo: 255,
                        flags: 4,
                        captured: 6,
                        moved: Piece::K as u8,
                    });
                }
                if (self.castling & 8) != 0
                    && (self.occ & (bb(59) | bb(58) | bb(57))) == 0
                    && !self.square_attacked(60, Side::White)
                    && !self.square_attacked(59, Side::White)
                    && !self.square_attacked(58, Side::White)
                {
                    list.push(Move {
                        from: 60,
                        to: 58,
                        promo: 255,
                        flags: 4,
                        captured: 6,
                        moved: Piece::K as u8,
                    });
                }
            }
            self.slide_add(Piece::B, &[-9, -7, 7, 9], occ_us, occ_them, list);
            self.slide_add(Piece::R, &[-8, -1, 1, 8], occ_us, occ_them, list);
            self.slide_add(
                Piece::Q,
                &[-9, -7, 7, 9, -8, -1, 1, 8],
                occ_us,
                occ_them,
                list,
            );
        }
        list.retain(|m| {
            let mut tmp = self.clone();
            tmp.make_move(*m);
            !tmp.in_check(self.stm)
        });
    }

    pub fn slide_add(
        &self,
        pc: Piece,
        deltas: &[i32],
        occ_us: u64,
        occ_them: u64,
        list: &mut Vec<Move>,
    ) {
        let mut b = self.bb_piece[self.stm.idx()][pc as usize];
        while b != 0 {
            let s = pop_lsb(&mut b);
            for &d in deltas {
                let mut t = s as i32 + d;
                while t >= 0 && t < 64 {
                    let tf = (t & 7) as i32;
                    let sf = (s & 7) as i32;
                    let tr = (t / 8) as i32;
                    let sr = (s / 8) as i32;
                    if (d == 1 || d == -1) && tr != sr {
                        break;
                    }
                    if (d == 9 || d == -9) && (tf - sf).abs() != (tr - sr).abs() {
                        break;
                    }
                    if (d == 7 || d == -7) && (tf - sf).abs() != (tr - sr).abs() {
                        break;
                    }
                    let tbb = bb(t as usize);
                    if (occ_us & tbb) != 0 {
                        break;
                    }
                    if (occ_them & tbb) != 0 {
                        let captured = self.piece_at(t as usize, self.stm.flip()).unwrap();
                        list.push(Move::capture(s, t as usize, captured as u8, pc as u8));
                        break;
                    } else {
                        list.push(Move::quiet(s, t as usize, pc as u8));
                    }
                    if (self.occ & tbb) != 0 {
                        break;
                    }
                    t += d;
                }
            }
        }
    }

    pub fn piece_at(&self, sq: usize, side: Side) -> Option<Piece> {
        for p in 0..6 {
            if (self.bb_piece[side.idx()][p] & bb(sq)) != 0 {
                return Some(match p {
                    0 => Piece::P,
                    1 => Piece::N,
                    2 => Piece::B,
                    3 => Piece::R,
                    4 => Piece::Q,
                    5 => Piece::K,
                    _ => unreachable!(),
                });
            }
        }
        None
    }

    pub fn make_move(&mut self, m: Move) {
        let us = self.stm.idx();
        let them = self.stm.flip().idx();
        self.hist.push(State {
            castling: self.castling,
            ep: self.ep,
            halfmove: self.halfmove,
            hash: self.hash,
        });
        self.move_hist.push(m);
        self.hash ^= unsafe {
            ZEP[self.ep as usize]
        };
        self.ep = 64;
        self.hash ^= unsafe {
            ZEP[self.ep as usize]
        };
        self.halfmove += 1;
        let from = m.from as usize;
        let to = m.to as usize;
        if m.flags & 1 != 0 {
            self.halfmove = 0;
            if m.flags & 2 != 0 {
                let cap_sq = if self.stm == Side::White {
                    to - 8
                } else {
                    to + 8
                };
                for p in 0..6 {
                    if (self.bb_piece[them][p] & bb(cap_sq)) != 0 {
                        self.bb_piece[them][p] ^= bb(cap_sq);
                        self.hash ^= unsafe {
                            ZP[them][p][cap_sq]
                        };
                        break;
                    }
                }
            } else {
                for p in 0..6 {
                    if (self.bb_piece[them][p] & bb(to)) != 0 {
                        self.bb_piece[them][p] ^= bb(to);
                        self.hash ^= unsafe {
                            ZP[them][p][to]
                        };
                        break;
                    }
                }
            }
        }
        let mut moved_p: usize = 6;
        for p in 0..6 {
            if (self.bb_piece[us][p] & bb(from)) != 0 {
                moved_p = p;
                self.bb_piece[us][p] ^= bb(from);
                self.hash ^= unsafe {
                    ZP[us][p][from]
                };
                break;
            }
        }
        if moved_p == 6 {
            panic!("no piece on from square");
        }
        if moved_p == Piece::P as usize {
            self.halfmove = 0;
        }
        if m.flags & 4 != 0 {
            if self.stm == Side::White {
                if to == 6 {
                    self.bb_piece[us][Piece::R as usize] ^= bb(7) | bb(5);
                    self.hash ^= unsafe {
                        ZP[us][Piece::R as usize][7] ^ ZP[us][Piece::R as usize][5]
                    };
                } else {
                    self.bb_piece[us][Piece::R as usize] ^= bb(0) | bb(3);
                    self.hash ^= unsafe {
                        ZP[us][Piece::R as usize][0] ^ ZP[us][Piece::R as usize][3]
                    };
                }
            } else {
                if to == 62 {
                    self.bb_piece[us][Piece::R as usize] ^= bb(63) | bb(61);
                    self.hash ^= unsafe {
                        ZP[us][Piece::R as usize][63] ^ ZP[us][Piece::R as usize][61]
                    };
                } else {
                    self.bb_piece[us][Piece::R as usize] ^= bb(56) | bb(59);
                    self.hash ^= unsafe {
                        ZP[us][Piece::R as usize][56] ^ ZP[us][Piece::R as usize][59]
                    };
                }
            }
        }
        if m.flags & 8 != 0 {
            self.hash ^= unsafe {
                ZEP[self.ep as usize]
            };
            self.ep = if self.stm == Side::White {
                (from + 8) as u8
            } else {
                (from - 8) as u8
            };
            self.hash ^= unsafe {
                ZEP[self.ep as usize]
            };
        }
        if m.promo != 255 {
            let pp = m.promo as usize;
            self.bb_piece[us][pp] ^= bb(to);
            self.hash ^= unsafe {
                ZP[us][pp][to]
            };
        } else {
            self.bb_piece[us][moved_p] ^= bb(to);
            self.hash ^= unsafe {
                ZP[us][moved_p][to]
            };
        }
        let old_castle = self.castling;
        let mut cr = self.castling;
        match from {
            4 => {
                if us == 0 {
                    cr &= !(1 | 2);
                }
            }
            60 => {
                if us == 1 {
                    cr &= !(4 | 8);
                }
            }
            0 => {
                cr &= !2;
            }
            7 => {
                cr &= !1;
            }
            56 => {
                cr &= !8;
            }
            63 => {
                cr &= !4;
            }
            _ => {}
        }
        match to {
            0 => {
                cr &= !2;
            }
            7 => {
                cr &= !1;
            }
            56 => {
                cr &= !8;
            }
            63 => {
                cr &= !4;
            }
            _ => {}
        }
        if cr != old_castle {
            self.hash ^= unsafe {
                ZCASTLE[old_castle as usize] ^ ZCASTLE[cr as usize]
            };
            self.castling = cr;
        }
        self.stm = self.stm.flip();
        self.hash ^= unsafe {
            ZSTM
        };
        self.compute_occ();
        if self.stm == Side::White {
            self.fullmove += 1;
        }
    }

    pub fn unmake(&mut self) {
        if let Some(st) = self.hist.pop() {
            let m = self.move_hist.pop().unwrap();
            self.castling = st.castling;
            self.ep = st.ep;
            self.halfmove = st.halfmove;
            self.hash = st.hash;
            self.stm = self.stm.flip();
            let us = self.stm.idx();
            let them = self.stm.flip().idx();

            let from = m.from as usize;
            let to = m.to as usize;

            if m.promo != 255 {
                self.bb_piece[us][Piece::P as usize] |= bb(from);
                self.bb_piece[us][m.promo as usize] &= !bb(to);
            } else {
                self.bb_piece[us][m.moved as usize] |= bb(from);
                self.bb_piece[us][m.moved as usize] &= !bb(to);
            }

            if m.flags & 1 != 0 {
                let cap_sq = if m.flags & 2 != 0 {
                    if self.stm == Side::White {
                        to - 8
                    } else {
                        to + 8
                    }
                } else {
                    to
                };
                self.bb_piece[them][m.captured as usize] |= bb(cap_sq);
            }

            if m.flags & 4 != 0 {
                if self.stm == Side::White {
                    if to == 6 {
                        self.bb_piece[us][Piece::R as usize] ^= bb(7) | bb(5);
                    } else {
                        self.bb_piece[us][Piece::R as usize] ^= bb(0) | bb(3);
                    }
                } else {
                    if to == 62 {
                        self.bb_piece[us][Piece::R as usize] ^= bb(63) | bb(61);
                    } else {
                        self.bb_piece[us][Piece::R as usize] ^= bb(56) | bb(59);
                    }
                }
            }
            if self.stm == Side::Black {
                self.fullmove -= 1;
            }
            self.compute_occ();
        }
    }
}
