use std::collections::HashMap;
use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use rand::Rng;

mod board;
mod perft;

use board::*;

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

const MAX_PLY: usize = 128;
const HISTORY_MAX: i32 = 1 << 16;
const ASPIRATION_WINDOW: i32 = 50;
const BISHOP_PAIR_BONUS: i32 = 40;
const ROOK_OPEN_BONUS: i32 = 25;
const ROOK_SEMI_OPEN_BONUS: i32 = 12;
const DOUBLED_PAWN_PENALTY: i32 = 12;
const ISOLATED_PAWN_PENALTY: i32 = 15;

const FILE_MASKS: [u64; 8] = [
    0x0101010101010101,
    0x0202020202020202,
    0x0404040404040404,
    0x0808080808080808,
    0x1010101010101010,
    0x2020202020202020,
    0x4040404040404040,
    0x8080808080808080,
];

type HistoryTable = [[[i32; 64]; 64]; 2];

const ADJACENT_FILES: [u64; 8] = [
    FILE_MASKS[1],
    FILE_MASKS[0] | FILE_MASKS[2],
    FILE_MASKS[1] | FILE_MASKS[3],
    FILE_MASKS[2] | FILE_MASKS[4],
    FILE_MASKS[3] | FILE_MASKS[5],
    FILE_MASKS[4] | FILE_MASKS[6],
    FILE_MASKS[5] | FILE_MASKS[7],
    FILE_MASKS[6],
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
    let white = Side::White as usize;
    let black = Side::Black as usize;
    if b.bb_piece[white][BISHOP].count_ones() >= 2 {
        s += BISHOP_PAIR_BONUS;
    }
    if b.bb_piece[black][BISHOP].count_ones() >= 2 {
        s -= BISHOP_PAIR_BONUS;
    }
    let white_pawns = b.bb_piece[white][PAWN];
    let black_pawns = b.bb_piece[black][PAWN];
    let white_rooks = b.bb_piece[white][ROOK];
    let black_rooks = b.bb_piece[black][ROOK];
    s += rook_file_score(white_rooks, white_pawns, black_pawns);
    s -= rook_file_score(black_rooks, black_pawns, white_pawns);
    s -= pawn_structure_penalty(white_pawns);
    s += pawn_structure_penalty(black_pawns);
    s
}
#[inline]
fn pst_val(p: usize, sq: usize) -> i32 {
    PST_W[p][sq]
}

#[derive(Clone)]
struct TTEntry {
    depth: i32,
    score: i32,
    flag: u8,
    best: Option<Move>,
}

struct Searcher {
    tt: Arc<Mutex<HashMap<u64, TTEntry>>>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    difficulty: Arc<Mutex<u8>>, // 1=Beginner, 2=Easy, 3=Medium, 4=Hard, 5=Expert
}

impl Searcher {
    fn new() -> Self {
        Self {
            tt: Arc::new(Mutex::new(HashMap::new())),
            stop: Arc::new(AtomicBool::new(false)),
            handle: None,
            difficulty: Arc::new(Mutex::new(5)), // Default to Expert level
        }
    }
}

struct SearchInstance {
    searcher: Searcher,
    nodes: u64,
    start: Instant,
    time_limit: Duration,
    killers: [[Option<Move>; 2]; MAX_PLY],
    history: HistoryTable,
}

impl SearchInstance {
    fn search(&mut self, b: &mut Board, depth: i32) -> (i32, Option<Move>) {
        let mut best = None;
        let mut score = 0;
        let mut prev_score = 0;
        'iter: for d in 1..=depth {
            if self.timed_out() {
                break;
            }
            let mut window = ASPIRATION_WINDOW;
            let mut alpha = -30000;
            let mut beta = 30000;
            let mut use_aspiration = d > 1;
            if use_aspiration {
                alpha = (prev_score - window).max(-30000);
                beta = (prev_score + window).min(30000);
            }
            loop {
                let (sc, bm) = self.alpha_beta(b, d, alpha, beta, 0);
                if self.searcher.stop.load(Ordering::Relaxed) {
                    break 'iter;
                }
                if use_aspiration && sc <= alpha {
                    window *= 2;
                    alpha = (prev_score - window).max(-30000);
                    beta = (prev_score + window).min(30000);
                    if alpha <= -30000 && beta >= 30000 {
                        use_aspiration = false;
                    }
                    continue;
                }
                if use_aspiration && sc >= beta {
                    window *= 2;
                    alpha = (prev_score - window).max(-30000);
                    beta = (prev_score + window).min(30000);
                    if alpha <= -30000 && beta >= 30000 {
                        use_aspiration = false;
                    }
                    continue;
                }
                score = sc;
                if bm.is_some() {
                    best = bm;
                }
                prev_score = sc;
                break;
            }
        }

        // Apply randomization for lower difficulty levels
        let difficulty = *self.searcher.difficulty.lock().unwrap();
        if difficulty < 3 && best.is_some() {
            best = self.maybe_randomize_move(b, best, difficulty);
        }

        (score, best)
    }

    fn maybe_randomize_move(&mut self, b: &mut Board, best: Option<Move>, difficulty: u8) -> Option<Move> {
        let mut rng = rand::thread_rng();

        // Determine randomization probability and pool size based on difficulty
        let (rand_prob, pool_size) = match difficulty {
            1 => (40, 5),  // Beginner: 40% chance, pick from top 5
            2 => (20, 3),  // Easy: 20% chance, pick from top 3
            _ => return best, // No randomization for medium and above
        };

        // Roll the dice
        if rng.gen_range(0..100) >= rand_prob {
            return best; // No randomization this time
        }

        // Generate and score all legal moves
        let mut moves = Vec::with_capacity(64);
        b.gen_moves(&mut moves);

        // Filter to legal moves only
        let mut legal_moves = Vec::new();
        for m in moves {
            b.make_move(m);
            if !b.in_check(b.stm.flip()) {
                legal_moves.push(m);
            }
            b.unmake();
        }

        if legal_moves.is_empty() {
            return best;
        }

        // Pick a random move from the pool
        let pick_from = pool_size.min(legal_moves.len());
        let idx = rng.gen_range(0..pick_from);
        Some(legal_moves[idx])
    }
    fn timed_out(&self) -> bool {
        self.start.elapsed() >= self.time_limit
    }
    fn probe(&self, key: u64) -> Option<TTEntry> {
        let tt = self.searcher.tt.lock().unwrap();
        tt.get(&key).cloned()
    }
    fn store(&mut self, key: u64, e: TTEntry) {
        let mut tt = self.searcher.tt.lock().unwrap();
        tt.insert(key, e);
    }
    fn alpha_beta(
        &mut self,
        b: &mut Board,
        depth: i32,
        mut alpha: i32,
        beta: i32,
        ply: usize,
    ) -> (i32, Option<Move>) {
        if self.timed_out() || self.searcher.stop.load(Ordering::Relaxed) {
            self.searcher.stop.store(true, Ordering::Relaxed);
            return (0, None);
        }
        self.nodes += 1;
        let stm = b.stm;
        let mut depth = depth;
        let in_check = b.in_check(b.stm);
        if in_check {
            depth += 1;
        }
        if depth <= 0 {
            return (self.qsearch(b, alpha, beta), None);
        }
        let static_eval = eval(b);
        let mut pv_move = None;
        if let Some(tt) = self.probe(b.hash) {
            if tt.depth >= depth {
                match tt.flag {
                    0 => return (tt.score, tt.best),
                    1 => {
                        if tt.score > alpha {
                            alpha = tt.score;
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
            pv_move = tt.best;
        }

        if depth >= 3
            && !in_check
            && ply < MAX_PLY - 1
            && has_non_king_material(b, stm)
            && static_eval >= beta
        {
            let reduction = 2;
            let r_depth = depth - 1 - reduction;
            if r_depth > 0 {
                let null_state = b.make_null_move();
                let score = -self.alpha_beta(b, r_depth, -beta, -beta + 1, ply + 1).0;
                b.unmake_null_move(null_state);
                if score >= beta {
                    return (beta, None);
                }
            }
        }

        let mut moves = Vec::with_capacity(64);
        b.gen_moves(&mut moves);
        if moves.is_empty() {
            if in_check {
                return (-29000 + ply as i32, None);
            }
            return (0, None);
        }

        let killers = if ply < MAX_PLY {
            self.killers[ply]
        } else {
            [None, None]
        };
        let board_ref: &Board = b;
        {
            let history = &self.history;
            moves.sort_by(|lhs, rhs| {
                move_score(board_ref, rhs, pv_move, &killers, stm, history)
                    .cmp(&move_score(board_ref, lhs, pv_move, &killers, stm, history))
            });
        }
        let mut best = None;
        let mut flag = 2;
        let mut legal_moves = 0;
        for (idx, m) in moves.into_iter().enumerate() {
            let is_quiet = (m.flags & FLAG_CAPTURE) == 0 && m.promo == 255;
            if !in_check && depth <= 2 && is_quiet {
                let margin = 100 * depth + 50;
                if static_eval + margin <= alpha {
                    continue;
                }
            }
            b.make_move(m);
            if b.in_check(b.stm.flip()) {
                b.unmake();
                continue;
            }
            legal_moves += 1;
            let gives_check = b.in_check(b.stm);
            let mut new_depth = depth - 1;
            if depth >= 3
                && idx >= 4
                && (m.flags & FLAG_CAPTURE) == 0
                && m.promo == 255
                && !gives_check
            {
                new_depth -= 1;
            }
            if new_depth < 0 {
                new_depth = 0;
            }
            let mut score;
            if best.is_none() {
                score = -self.alpha_beta(b, new_depth, -beta, -alpha, ply + 1).0;
            } else {
                score = -self.alpha_beta(b, new_depth, -alpha - 1, -alpha, ply + 1).0;
                if score > alpha && score < beta {
                    score = -self.alpha_beta(b, new_depth, -beta, -alpha, ply + 1).0;
                }
            }
            b.unmake();
            if score > alpha {
                alpha = score;
                best = Some(m);
                flag = 0;
                if is_quiet {
                    self.update_history(stm, m, depth);
                }
                if alpha >= beta {
                    if (m.flags & FLAG_CAPTURE) == 0 && m.promo == 255 {
                        self.store_killer(ply, m);
                    }
                    flag = 1;
                    self.store(
                        b.hash,
                        TTEntry {
                            depth,
                            score: alpha,
                            flag,
                            best,
                        },
                    );
                    return (alpha, best);
                }
            }
        }
        if legal_moves == 0 {
            if in_check {
                return (-29000 + ply as i32, None);
            }
            return (0, None);
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
        if self.timed_out() || self.searcher.stop.load(Ordering::Relaxed) {
            self.searcher.stop.store(true, Ordering::Relaxed);
            return 0;
        }
        let stand = eval(b);
        if stand >= beta {
            return beta;
        }
        const DELTA_PRUNE: i32 = 200;
        if stand + DELTA_PRUNE < alpha {
            return alpha;
        }
        if stand > alpha {
            alpha = stand
        }
        let mut moves = Vec::with_capacity(32);
        b.gen_moves(&mut moves);
        for m in moves {
            let is_capture = (m.flags & FLAG_CAPTURE) != 0;
            let is_promo = m.promo != 255;
            b.make_move(m);
            if b.in_check(b.stm.flip()) {
                b.unmake();
                continue;
            }
            let gives_check = b.in_check(b.stm);
            if !is_capture && !is_promo && !gives_check {
                b.unmake();
                continue;
            }
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

    fn store_killer(&mut self, ply: usize, m: Move) {
        if ply >= MAX_PLY {
            return;
        }
        if self.killers[ply][0] != Some(m) {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = Some(m);
        }
    }

    fn update_history(&mut self, side: Side, m: Move, depth: i32) {
        if depth <= 0 {
            return;
        }
        let idx = side as usize;
        let from = m.from as usize;
        let to = m.to as usize;
        let bonus = depth * depth;
        let entry = &mut self.history[idx][from][to];
        *entry += bonus;
        if *entry > HISTORY_MAX {
            *entry = HISTORY_MAX;
        }
    }
}

fn move_score(
    board: &Board,
    m: &Move,
    pv_move: Option<Move>,
    killers: &[Option<Move>; 2],
    stm: Side,
    history: &HistoryTable,
) -> i32 {
    if let Some(pv) = pv_move {
        if pv == *m {
            return 1_000_000;
        }
    }
    if killers.iter().any(|&k| k == Some(*m)) {
        return 900_000;
    }
    if (m.flags & FLAG_CAPTURE) != 0 {
        return 800_000 + mvv_lva(board, m, stm);
    }
    if m.promo != 255 {
        return 700_000;
    }
    500_000 + history[stm as usize][m.from as usize][m.to as usize]
}

fn mvv_lva(b: &Board, m: &Move, stm: Side) -> i32 {
    let victim = b
        .piece_at(stm.flip(), m.to as usize)
        .map(|p| MVAL[p])
        .unwrap_or(0);
    let attacker = b
        .piece_at(stm, m.from as usize)
        .map(|p| MVAL[p])
        .unwrap_or(0);
    victim * 10 - attacker
}

fn has_non_king_material(b: &Board, side: Side) -> bool {
    let mut occ = b.bb_side[side as usize];
    occ &= !b.bb_piece[side as usize][KING];
    occ != 0
}

fn rook_file_score(rooks: u64, own_pawns: u64, opp_pawns: u64) -> i32 {
    let mut score = 0;
    for file in 0..8 {
        let mask = FILE_MASKS[file];
        if rooks & mask == 0 {
            continue;
        }
        let own = own_pawns & mask != 0;
        let opp = opp_pawns & mask != 0;
        if !own && !opp {
            score += ROOK_OPEN_BONUS;
        } else if !own && opp {
            score += ROOK_SEMI_OPEN_BONUS;
        }
    }
    score
}

fn pawn_structure_penalty(pawns: u64) -> i32 {
    let mut penalty = 0;
    for file in 0..8 {
        let mask = FILE_MASKS[file];
        let count = (pawns & mask).count_ones() as i32;
        if count > 1 {
            penalty += DOUBLED_PAWN_PENALTY * (count - 1);
        }
        if count > 0 {
            if (pawns & ADJACENT_FILES[file]) == 0 {
                penalty += ISOLATED_PAWN_PENALTY;
            }
        }
    }
    penalty
}

fn main() {
    init_tables();
    init_zobrist();
    let stdin = io::stdin();
    let mut b = Board::startpos();
    let mut searcher = Searcher::new();
    let mut xboard_mode = false;
    for line in stdin.lock().lines() {
        if let Ok(line) = line {
            handle_line(line.trim(), &mut b, &mut searcher, &mut xboard_mode);
        } else {
            break;
        }
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
        println!("option name Skill Level type spin default 5 min 1 max 5");
        println!("uciok");
        return;
    }
    if line == "isready" {
        println!("readyok");
        return;
    }
    if line == "ucinewgame" {
        *b = Board::startpos();
        s.tt.lock().unwrap().clear();
        println!("readyok");
        return;
    }
    if line == "stop" {
        s.stop.store(true, Ordering::Relaxed);
        if let Some(h) = s.handle.take() {
            h.join().unwrap();
        }
        return;
    }
    if line.starts_with("setoption") {
        handle_setoption(line, s);
        return;
    }
    if line.starts_with("position ") {
        set_position(line, b);
        return;
    }
    if line.starts_with("go") {
        let params = parse_go(line);
        let difficulty = *s.difficulty.lock().unwrap();
        let depth = get_depth_for_difficulty(difficulty, params.depth);
        let time_ms = time_for_move(b.stm, &params);
        let mut b2 = b.clone();
        let tt = s.tt.clone();
        let stop = s.stop.clone();
        let diff = s.difficulty.clone();
        s.handle = Some(std::thread::spawn(move || {
            stop.store(false, Ordering::Relaxed);
            let mut si = SearchInstance {
                searcher: Searcher {
                    tt,
                    stop,
                    handle: None,
                    difficulty: diff,
                },
                nodes: 0,
                start: Instant::now(),
                time_limit: Duration::from_millis(time_ms),
                killers: [[None; 2]; MAX_PLY],
                history: [[[0; 64]; 64]; 2],
            };
            let (score, mv) = si.search(&mut b2, depth);
            if let Some(m) = mv {
                println!("info depth {} score cp {} nodes {}", depth, score, si.nodes);
                println!("bestmove {}", move_to_str(m));
            } else {
                println!("bestmove 0000");
            }
        }));
        return;
    }
    if line.starts_with("perft ") {
        let d = line
            .split_whitespace()
            .nth(1)
            .and_then(|x| x.parse::<u32>().ok())
            .unwrap_or(3);
        perft::test(b, d);
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
        s.tt.lock().unwrap().clear();
        return;
    }
    if line.starts_with("setboard ") {
        xboard_set_position(line, b);
        return;
    }
    if line.starts_with("go") || line == "go" {
        let difficulty = *s.difficulty.lock().unwrap();
        let depth = get_depth_for_difficulty(difficulty, None);
        let mut b2 = b.clone();
        let tt = s.tt.clone();
        let stop = s.stop.clone();
        let diff = s.difficulty.clone();
        s.handle = Some(std::thread::spawn(move || {
            stop.store(false, Ordering::Relaxed);
            let mut si = SearchInstance {
                searcher: Searcher {
                    tt,
                    stop,
                    handle: None,
                    difficulty: diff,
                },
                nodes: 0,
                start: Instant::now(),
                time_limit: Duration::from_millis(5000),
                killers: [[None; 2]; MAX_PLY],
                history: [[[0; 64]; 64]; 2],
            };
            let (_score, mv) = si.search(&mut b2, depth);
            if let Some(m) = mv {
                println!("move {}", move_to_str(m));
            }
        }));
        return;
    }
    if line.starts_with("move ") {
        xboard_make_move(line, b);
        return;
    }
    if line.starts_with("hint") {
        let difficulty = *s.difficulty.lock().unwrap();
        let depth = get_depth_for_difficulty(difficulty, Some(4));
        let mut b2 = b.clone();
        let tt = s.tt.clone();
        let stop = s.stop.clone();
        let diff = s.difficulty.clone();
        s.handle = Some(std::thread::spawn(move || {
            stop.store(false, Ordering::Relaxed);
            let mut si = SearchInstance {
                searcher: Searcher {
                    tt,
                    stop,
                    handle: None,
                    difficulty: diff,
                },
                nodes: 0,
                start: Instant::now(),
                time_limit: Duration::from_millis(2000),
                killers: [[None; 2]; MAX_PLY],
                history: [[[0; 64]; 64]; 2],
            };
            let (_score, mv) = si.search(&mut b2, depth);
            if let Some(m) = mv {
                println!("Hint: {}", move_to_str(m));
            }
        }));
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
    }
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

fn handle_setoption(cmd: &str, s: &mut Searcher) {
    // Parse: setoption name <name> value <value>
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    let mut name_idx = None;
    let mut value_idx = None;

    for (i, &part) in parts.iter().enumerate() {
        if part == "name" && i + 1 < parts.len() {
            name_idx = Some(i + 1);
        }
        if part == "value" && i + 1 < parts.len() {
            value_idx = Some(i + 1);
        }
    }

    if let (Some(n_idx), Some(v_idx)) = (name_idx, value_idx) {
        let name = parts[n_idx..v_idx.min(parts.len()).saturating_sub(1)].join(" ").to_lowercase();

        if name.contains("skill") || name.contains("difficulty") {
            if let Ok(level) = parts[v_idx].parse::<u8>() {
                let clamped = level.clamp(1, 5);
                *s.difficulty.lock().unwrap() = clamped;
                println!("info string Difficulty level set to {}", clamped);
            }
        }
    }
}

fn get_depth_for_difficulty(difficulty: u8, requested_depth: Option<i32>) -> i32 {
    // If depth is explicitly requested, respect it for Expert level only
    if difficulty == 5 {
        return requested_depth.unwrap_or(100);
    }

    // Otherwise use difficulty-based depth
    match difficulty {
        1 => 2,  // Beginner
        2 => 3,  // Easy
        3 => 5,  // Medium
        4 => 7,  // Hard
        _ => requested_depth.unwrap_or(100), // Expert (default)
    }
}

fn time_for_move(stm: Side, params: &GoParams) -> u64 {
    if let Some(mt) = params.movetime {
        return mt;
    }
    let (time, inc) = if stm == Side::White {
        (params.wtime, params.winc)
    } else {
        (params.btime, params.binc)
    };
    let moves_to_go = params.movestogo.unwrap_or(60).max(1);
    match time {
        Some(t) => {
            let base = t / moves_to_go;
            let bonus = inc.unwrap_or(0) / 2;
            let slice = base + bonus;
            let max_allowed = t.saturating_sub(50).max(50);
            slice.max(50).min(max_allowed)
        }
        None => 2000,
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

#[derive(Default)]
struct GoParams {
    wtime: Option<u64>,
    btime: Option<u64>,
    winc: Option<u64>,
    binc: Option<u64>,
    movetime: Option<u64>,
    movestogo: Option<u64>,
    depth: Option<i32>,
}

fn parse_go(s: &str) -> GoParams {
    let mut params = GoParams::default();
    let mut it = s.split_whitespace();
    it.next();
    while let Some(tok) = it.next() {
        match tok {
            "wtime" => params.wtime = it.next().and_then(|x| x.parse().ok()),
            "btime" => params.btime = it.next().and_then(|x| x.parse().ok()),
            "winc" => params.winc = it.next().and_then(|x| x.parse().ok()),
            "binc" => params.binc = it.next().and_then(|x| x.parse().ok()),
            "movetime" => params.movetime = it.next().and_then(|x| x.parse().ok()),
            "movestogo" => params.movestogo = it.next().and_then(|x| x.parse().ok()),
            "depth" => params.depth = it.next().and_then(|x| x.parse().ok()),
            _ => {}
        }
    }
    params
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
