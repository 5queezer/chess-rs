// Rust Chess Engine — single-file prototype
use std::collections::HashMap;
use std::io::{self, Read};
use std::time::{Duration, Instant};

#[inline]
fn bb(sq: usize) -> u64 {
    1u64 << sq
}
#[inline]
fn lsb(x: u64) -> usize {
    x.trailing_zeros() as usize
}
#[inline]
fn pop_lsb(x: &mut u64) -> usize {
    let s = lsb(*x);
    *x &= *x - 1;
    s
}

const FILE_A: u64 = 0x0101010101010101;
const FILE_H: u64 = 0x8080808080808080;
const RANK_3: u64 = 0x0000000000FF0000;
const RANK_6: u64 = 0x0000FF0000000000;

fn sq_from_str(s: &str) -> Option<usize> {
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

fn sq_to_str(sq: usize) -> String {
    format!(
        "{}{}",
        (b'a' + (sq & 7) as u8) as char,
        (b'1' + (sq / 8) as u8) as char
    )
}

static mut KNIGHT: [u64; 64] = [0; 64];
static mut KING: [u64; 64] = [0; 64];
static mut PAWN_ATK: [[u64; 64]; 2] = [[0; 64], [0; 64]];

fn init_tables() {
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    White = 0,
    Black = 1,
}
impl Side {
    fn idx(self) -> usize {
        if self == Side::White { 0 } else { 1 }
    }
    fn flip(self) -> Side {
        if self == Side::White {
            Side::Black
        } else {
            Side::White
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Piece {
    P,
    N,
    B,
    R,
    Q,
    K,
}

#[derive(Clone, Copy)]
struct Move {
    from: u8,
    to: u8,
    promo: u8,
    flags: u8,
}
impl Move {
    fn quiet(from: usize, to: usize) -> Self {
        Self {
            from: from as u8,
            to: to as u8,
            promo: 255,
            flags: 0,
        }
    }
    fn capture(from: usize, to: usize) -> Self {
        Self {
            from: from as u8,
            to: to as u8,
            promo: 255,
            flags: 1,
        }
    }
    fn promo(from: usize, to: usize, p: u8, cap: bool) -> Self {
        Self {
            from: from as u8,
            to: to as u8,
            promo: p,
            flags: if cap { 1 } else { 0 },
        }
    }
}

#[derive(Clone)]
struct State {
    castling: u8,
    ep: u8,
    halfmove: u16,
    hash: u64,
}

#[derive(Clone)]
struct Board {
    bb_side: [u64; 2],
    bb_piece: [[u64; 6]; 2],
    occ: u64,
    stm: Side,
    castling: u8,
    ep: u8,
    halfmove: u16,
    fullmove: u16,
    hash: u64,
    hist: Vec<State>,
}

static mut ZP: [[[u64; 64]; 6]; 2] = [[[0; 64]; 6]; 2];
static mut ZCASTLE: [u64; 16] = [0; 16];
static mut ZEP: [u64; 65] = [0; 65];
static mut ZSTM: u64 = 0;

fn rng64(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    *seed
}

fn init_zobrist() {
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
    fn empty() -> Self {
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
        }
    }
    fn compute_occ(&mut self) {
        self.bb_side[0] = 0;
        self.bb_side[1] = 0;
        for s in 0..2 {
            for p in 0..6 {
                self.bb_side[s] |= self.bb_piece[s][p];
            }
        }
        self.occ = self.bb_side[0] | self.bb_side[1];
    }
    fn update_hash(&mut self) {
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

    fn from_fen(fen: &str) -> Self {
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

    fn startpos() -> Self {
        Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    }

    fn in_check(&self, side: Side) -> bool {
        let king_sq = lsb(self.bb_piece[side.idx()][Piece::K as usize]);
        self.square_attacked(king_sq, side.flip())
    }

    fn square_attacked(&self, sq: usize, by: Side) -> bool {
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

    fn ray_attacks(&self, from: usize, deltas: &[i32]) -> u64 {
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

    fn gen_moves(&self, list: &mut Vec<Move>) {
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
                            list.push(Move::promo(from, to, p as u8, false));
                        }
                    } else {
                        list.push(Move::quiet(from, to));
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
                    });
                }
                let left = ((pawns & !FILE_A) << 7) & occ_them;
                let right = ((pawns & !FILE_H) << 9) & occ_them;
                let mut c = left;
                while c != 0 {
                    let to = pop_lsb(&mut c);
                    let from = to - 7;
                    if to >= 56 {
                        for &p in &[Piece::N, Piece::B, Piece::R, Piece::Q] {
                            list.push(Move::promo(from, to, p as u8, true));
                        }
                    } else {
                        list.push(Move::capture(from, to));
                    }
                }
                let mut c2 = right;
                while c2 != 0 {
                    let to = pop_lsb(&mut c2);
                    let from = to - 9;
                    if to >= 56 {
                        for &p in &[Piece::N, Piece::B, Piece::R, Piece::Q] {
                            list.push(Move::promo(from, to, p as u8, true));
                        }
                    } else {
                        list.push(Move::capture(from, to));
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
                            list.push(Move::promo(from, to, p as u8, false));
                        }
                    } else {
                        list.push(Move::quiet(from, to));
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
                    });
                }
                let left = ((pawns & !FILE_H) >> 7) & occ_them;
                let right = ((pawns & !FILE_A) >> 9) & occ_them;
                let mut c = left;
                while c != 0 {
                    let to = pop_lsb(&mut c);
                    let from = to + 7;
                    if to < 8 {
                        for &p in &[Piece::N, Piece::B, Piece::R, Piece::Q] {
                            list.push(Move::promo(from, to, p as u8, true));
                        }
                    } else {
                        list.push(Move::capture(from, to));
                    }
                }
                let mut c2 = right;
                while c2 != 0 {
                    let to = pop_lsb(&mut c2);
                    let from = to + 9;
                    if to < 8 {
                        for &p in &[Piece::N, Piece::B, Piece::R, Piece::Q] {
                            list.push(Move::promo(from, to, p as u8, true));
                        }
                    } else {
                        list.push(Move::capture(from, to));
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
                        list.push(Move::capture(s, t));
                    } else {
                        list.push(Move::quiet(s, t));
                    }
                }
            }
            let ksq = lsb(self.bb_piece[us][Piece::K as usize]);
            let mut km = KING[ksq] & !occ_us;
            while km != 0 {
                let t = pop_lsb(&mut km);
                if (occ_them & bb(t)) != 0 {
                    list.push(Move::capture(ksq, t));
                } else {
                    list.push(Move::quiet(ksq, t));
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

    fn slide_add(
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
                        list.push(Move::capture(s, t as usize));
                        break;
                    } else {
                        list.push(Move::quiet(s, t as usize));
                    }
                    if (self.occ & tbb) != 0 {
                        break;
                    }
                    t += d;
                }
            }
        }
    }

    fn make_move(&mut self, m: Move) {
        let us = self.stm.idx();
        let them = self.stm.flip().idx();
        self.hist.push(State {
            castling: self.castling,
            ep: self.ep,
            halfmove: self.halfmove,
            hash: self.hash,
        });
        self.hash ^= unsafe { ZEP[self.ep as usize] };
        self.ep = 64;
        self.hash ^= unsafe { ZEP[self.ep as usize] };
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
                        self.hash ^= unsafe { ZP[them][p][cap_sq] };
                        break;
                    }
                }
            } else {
                for p in 0..6 {
                    if (self.bb_piece[them][p] & bb(to)) != 0 {
                        self.bb_piece[them][p] ^= bb(to);
                        self.hash ^= unsafe { ZP[them][p][to] };
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
                self.hash ^= unsafe { ZP[us][p][from] };
                break;
            }
        }
        if moved_p == Piece::P as usize {
            self.halfmove = 0;
        }
        if m.flags & 4 != 0 {
            if self.stm == Side::White {
                if to == 6 {
                    self.bb_piece[us][Piece::R as usize] ^= bb(7) | bb(5);
                    self.hash ^=
                        unsafe { ZP[us][Piece::R as usize][7] ^ ZP[us][Piece::R as usize][5] };
                } else {
                    self.bb_piece[us][Piece::R as usize] ^= bb(0) | bb(3);
                    self.hash ^=
                        unsafe { ZP[us][Piece::R as usize][0] ^ ZP[us][Piece::R as usize][3] };
                }
            } else {
                if to == 62 {
                    self.bb_piece[us][Piece::R as usize] ^= bb(63) | bb(61);
                    self.hash ^=
                        unsafe { ZP[us][Piece::R as usize][63] ^ ZP[us][Piece::R as usize][61] };
                } else {
                    self.bb_piece[us][Piece::R as usize] ^= bb(56) | bb(59);
                    self.hash ^=
                        unsafe { ZP[us][Piece::R as usize][56] ^ ZP[us][Piece::R as usize][59] };
                }
            }
        }
        if m.flags & 8 != 0 {
            self.hash ^= unsafe { ZEP[self.ep as usize] };
            self.ep = if self.stm == Side::White {
                (from + 8) as u8
            } else {
                (from - 8) as u8
            };
            self.hash ^= unsafe { ZEP[self.ep as usize] };
        }
        if m.promo != 255 {
            let pp = m.promo as usize;
            self.bb_piece[us][pp] ^= bb(to);
            self.hash ^= unsafe { ZP[us][pp][to] };
        } else {
            self.bb_piece[us][moved_p] ^= bb(to);
            self.hash ^= unsafe { ZP[us][moved_p][to] };
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
            self.hash ^= unsafe { ZCASTLE[old_castle as usize] ^ ZCASTLE[cr as usize] };
            self.castling = cr;
        }
        self.stm = self.stm.flip();
        self.hash ^= unsafe { ZSTM };
        self.compute_occ();
        if self.stm == Side::White {
            self.fullmove += 1;
        }
    }

    fn unmake(&mut self) {
        if let Some(st) = self.hist.pop() {
            self.castling = st.castling;
            self.ep = st.ep;
            self.halfmove = st.halfmove;
            self.hash = st.hash;
            self.stm = self.stm.flip();
            self.compute_occ();
        }
    }
}

const MVAL: [i32; 6] = [100, 320, 330, 500, 900, 0];
static PST_W: [[i32; 64]; 6] = [
    [
        0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, -20, -20, 10, 10, 5, 5, -5, -10, 0, 0, -10, -5, 5, 0, 0,
        0, 20, 20, 0, 0, 0, 5, 5, 10, 25, 25, 10, 5, 5, 10, 10, 20, 30, 30, 20, 10, 10, 50, 50, 50,
        50, 50, 50, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 0, 0, 0, -20, -40, -30, 0, 10, 15, 15,
        10, 0, -30, -30, 5, 15, 20, 20, 15, 5, -30, -30, 0, 15, 20, 20, 15, 0, -30, -30, 5, 10, 15,
        15, 10, 5, -30, -40, -20, 0, 5, 5, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50,
    ],
    [
        -20, -10, -10, -10, -10, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 10, 10, 5,
        0, -10, -10, 5, 5, 10, 10, 5, 5, -10, -10, 0, 10, 10, 10, 10, 0, -10, -10, 10, 10, 10, 10,
        10, 10, -10, -10, 5, 0, 0, 0, 0, 5, -10, -20, -10, -10, -10, -10, -10, -10, -20,
    ],
    [
        0, 0, 0, 5, 5, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0,
        0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, 5, 10, 10, 10, 10, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        -20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 5, 5, 5, 0,
        -10, -5, 0, 5, 5, 5, 5, 0, -5, 0, 0, 5, 5, 5, 5, 0, -5, -10, 5, 5, 5, 5, 5, 0, -10, -10, 0,
        5, 0, 0, 0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20,
    ],
    [
        -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40,
        -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -20, -30, -30, -40,
        -40, -30, -30, -20, -10, -20, -20, -20, -20, -20, -20, -10, 20, 20, 0, 0, 0, 0, 20, 20, 20,
        30, 10, 0, 0, 10, 30, 20,
    ],
];

fn eval(b: &Board) -> i32 {
    let mut s = 0;
    for c in 0..2 {
        for p in 0..6 {
            let mut bbp = b.bb_piece[c][p];
            while bbp != 0 {
                let sq = pop_lsb(&mut bbp);
                let v = MVAL[p] + pst_val(p, if c == 0 { sq } else { 63 - sq });
                s += if c == 0 { v } else { -v };
            }
        }
    }
    s
}
#[inline]
fn pst_val(p: usize, sq: usize) -> i32 {
    PST_W[p][sq]
}

struct TTEntry {
    depth: i32,
    score: i32,
    flag: u8,
    best: Option<Move>,
}
struct Searcher {
    tt: HashMap<u64, TTEntry>,
    nodes: u64,
    start: Instant,
    time_limit: Duration,
}

impl Searcher {
    fn new() -> Self {
        Self {
            tt: HashMap::new(),
            nodes: 0,
            start: Instant::now(),
            time_limit: Duration::from_secs(3600),
        }
    }
    fn search(&mut self, b: &mut Board, depth: i32, time_ms: u64) -> (i32, Option<Move>) {
        self.start = Instant::now();
        self.time_limit = Duration::from_millis(time_ms);
        let mut best = None;
        let mut score = 0;
        for d in 1..=depth {
            let (sc, bm) = self.alpha_beta(b, d, -30000, 30000);
            score = sc;
            if bm.is_some() {
                best = bm;
            }
            if self.timed_out() {
                break;
            }
        }
        (score, best)
    }
    fn timed_out(&self) -> bool {
        self.start.elapsed() >= self.time_limit
    }
    fn probe(&self, key: u64) -> Option<&TTEntry> {
        self.tt.get(&key)
    }
    fn store(&mut self, key: u64, e: TTEntry) {
        self.tt.insert(key, e);
    }
    fn alpha_beta(
        &mut self,
        b: &mut Board,
        depth: i32,
        mut alpha: i32,
        beta: i32,
    ) -> (i32, Option<Move>) {
        if self.timed_out() {
            return (0, None);
        }
        self.nodes += 1;
        if depth == 0 {
            return (self.qsearch(b, alpha, beta), None);
        }
        if let Some(tt) = self.probe(b.hash) {
            if tt.depth >= depth {
                match tt.flag {
                    0 => return (tt.score, tt.best),
                    1 => {
                        if tt.score > alpha {
                            alpha = tt.score
                        }
                    }
                    2 => {
                        if tt.score < beta {
                            return (tt.score, tt.best);
                        }
                    }
                    _ => {}
                }
                if alpha >= beta {
                    return (tt.score, tt.best);
                }
            }
        }
        let mut moves = Vec::with_capacity(64);
        b.gen_moves(&mut moves);
        if moves.is_empty() {
            if b.in_check(b.stm) {
                return (-29000 + (5 - depth), None);
            } else {
                return (0, None);
            }
        }
        if let Some(tt) = self.probe(b.hash) {
            if let Some(pv) = tt.best {
                if let Some(i) = moves.iter().position(|m| {
                    m.from == pv.from && m.to == pv.to && m.promo == pv.promo && m.flags == pv.flags
                }) {
                    moves.swap(0, i);
                }
            }
        }
        moves.sort_by_key(|m| ((m.flags & 1) != 0) as i32 * 1000 + (m.promo != 255) as i32 * 500);
        moves.reverse();
        let mut best = None;
        let mut flag = 2;
        for m in moves {
            b.make_move(m);
            let (sc, _) = self.alpha_beta(b, depth - 1, -beta, -alpha);
            let sc = -sc;
            b.unmake();
            if sc > alpha {
                alpha = sc;
                best = Some(m);
                flag = 0;
                if alpha >= beta {
                    flag = 1;
                    break;
                }
            }
        }
        self.store(
            b.hash,
            TTEntry {
                depth,
                score: alpha,
                flag,
                best,
            },
        );
        (alpha, best)
    }
    fn qsearch(&mut self, b: &mut Board, mut alpha: i32, beta: i32) -> i32 {
        let stand = eval(b);
        if stand >= beta {
            return beta;
        }
        if stand > alpha {
            alpha = stand
        }
        let mut moves = Vec::with_capacity(32);
        b.gen_moves(&mut moves);
        moves.retain(|m| m.flags & 1 != 0 || m.promo != 255);
        for m in moves {
            b.make_move(m);
            let sc = -self.qsearch(b, -beta, -alpha);
            b.unmake();
            if sc >= beta {
                return beta;
            }
            if sc > alpha {
                alpha = sc
            }
        }
        alpha
    }
}

fn perft(b: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut moves = Vec::new();
    b.gen_moves(&mut moves);
    if depth == 1 {
        return moves.len() as u64;
    }
    let mut nodes = 0;
    for m in moves {
        b.make_move(m);
        nodes += perft(b, depth - 1);
        b.unmake();
    }
    nodes
}

fn main() {
    init_tables();
    init_zobrist();
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let mut b = Board::startpos();
    let mut searcher = Searcher::new();
    let mut xboard_mode = false;
    for line in input.lines() {
        handle_line(line.trim(), &mut b, &mut searcher, &mut xboard_mode);
    }
}

fn handle_line(line: &str, b: &mut Board, s: &mut Searcher, xboard_mode: &mut bool) {
    let line = line.trim();
    if line.is_empty() {
        return;
    }

    // UCI protocol
    if line == "uci" {
        println!("id name rce");
        println!("id author openai");
        println!("uciok");
        return;
    }
    if line == "isready" {
        println!("readyok");
        return;
    }
    if line == "ucinewgame" {
        *b = Board::startpos();
        s.tt.clear();
        println!("readyok");
        return;
    }
    if line.starts_with("position ") {
        set_position(line, b);
        return;
    }
    if line.starts_with("go") {
        let depth = parse_depth(line).unwrap_or(6);
        let (score, mv) = s.search(b, depth, 5000);
        if let Some(m) = mv {
            println!("info depth {} score cp {} nodes {}", depth, score, s.nodes);
            println!("bestmove {}", move_to_str(m));
        } else {
            println!("bestmove 0000");
        }
        return;
    }
    if line.starts_with("perft ") {
        let d = line
            .split_whitespace()
            .nth(1)
            .and_then(|x| x.parse::<u32>().ok())
            .unwrap_or(3);
        let n = perft(b, d);
        println!("nodes {}", n);
        return;
    }

    // xboard/GNU Chess protocol
    if line == "xboard" {
        *xboard_mode = true;
        return;
    }
    if line == "protover" {
        println!("feature myname=\"rce\" variants=\"normal\" colors=0 time=1 sigint=0");
        return;
    }
    if line == "new" {
        *b = Board::startpos();
        s.tt.clear();
        return;
    }
    if line.starts_with("setboard ") {
        xboard_set_position(line, b);
        return;
    }
    if line.starts_with("go") || line == "go" {
        let depth = 6;
        let (_score, mv) = s.search(b, depth, 5000);
        if let Some(m) = mv {
            println!("move {}", move_to_str(m));
        }
        return;
    }
    if line.starts_with("move ") {
        xboard_make_move(line, b);
        return;
    }
    if line.starts_with("hint") {
        let depth = 4;
        let (_score, mv) = s.search(b, depth, 2000);
        if let Some(m) = mv {
            println!("Hint: {}", move_to_str(m));
        }
        return;
    }
    if line == "force" {
        return;
    } // Stop searching (not implemented in detail)
    if line == "random" {
        return;
    }
    if line.starts_with("level") {
        return;
    } // Time controls (not implemented in detail)
    if line.starts_with("time") {
        return;
    }
    if line.starts_with("otim") {
        return;
    }
    if line == "remove" {
        if b.hist.len() > 0 {
            b.unmake();
        }
        if b.hist.len() > 0 {
            b.unmake();
        }
        return;
    } // Undo last two half-moves
    if line == "undo" {
        if b.hist.len() > 0 {
            b.unmake();
        }
        return;
    }

    if line == "quit" {
        return;
    }
}

fn xboard_set_position(cmd: &str, b: &mut Board) {
    let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
    if parts.len() == 2 {
        *b = Board::from_fen(parts[1]);
    }
}

fn xboard_make_move(cmd: &str, b: &mut Board) {
    let move_str = cmd.strip_prefix("move ").unwrap_or("");
    if let Some(m) = str_to_move(b, move_str) {
        b.make_move(m);
    }
}

fn parse_depth(s: &str) -> Option<i32> {
    let mut it = s.split_whitespace();
    it.next();
    while let Some(tok) = it.next() {
        if tok == "depth" {
            return it.next().and_then(|x| x.parse().ok());
        }
    }
    None
}

fn set_position(cmd: &str, b: &mut Board) {
    let mut it = cmd.split_whitespace();
    it.next();
    match it.next() {
        Some("startpos") => {
            *b = Board::startpos();
        }
        Some("fen") => {
            let fen: Vec<&str> = it.clone().take(6).collect();
            *b = Board::from_fen(&fen.join(" "));
            for _ in 0..6 {
                it.next();
            }
        }
        _ => {}
    }
    if let Some("moves") = it.next() {
        for mv in it {
            if let Some(m) = str_to_move(b, mv) {
                b.make_move(m);
            }
        }
    }
}

fn move_to_str(m: Move) -> String {
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

fn str_to_move(b: &Board, s: &str) -> Option<Move> {
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
