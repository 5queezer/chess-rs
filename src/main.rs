use std::collections::HashMap;
use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use rand::Rng;

mod board;
mod perft;
mod ml;

use board::*;
use ml::MLEvaluator;

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

// Performance tuning constants
const MAX_MOVE_TIME_MS: u64 = 60000; // Hard limit: 60 seconds per move
const MIN_MOVE_TIME_MS: u64 = 50;    // Minimum time to think
const TIME_SAFETY_MARGIN_MS: u64 = 100; // Reserve time to avoid timeouts
const ML_TIME_MULTIPLIER: f64 = 0.3; // Reduce time allocation when ML is enabled (ML is ~10x slower)

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

/// Classical hand-crafted evaluation function
fn eval_classical(b: &Board) -> i32 {
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
    ml_evaluator: Arc<Mutex<Option<MLEvaluator>>>,
    use_ml: Arc<Mutex<bool>>,
    // XBoard protocol state
    force_mode: bool,
    post_mode: bool,
    engine_color: Option<Side>,
    xboard_time: i32,        // Engine's remaining time in centiseconds
    xboard_otim: i32,        // Opponent's time in centiseconds
    time_per_move: Option<i32>, // Fixed time per move in centiseconds (st command)
    depth_limit: Option<i32>,   // Fixed depth limit (sd command)
    level_moves: i32,        // Moves per time control (0 = game in X)
    level_base: i32,         // Base time in centiseconds
    level_inc: i32,          // Increment in centiseconds
}

impl Searcher {
    fn new() -> Self {
        // Try to initialize ML evaluator
        let ml_eval = match MLEvaluator::new() {
            Ok(evaluator) => {
                evaluator.print_status();
                Some(evaluator)
            }
            Err(e) => {
                eprintln!("⚠️  ML initialization failed: {}", e);
                eprintln!("   Using classical evaluation only");
                None
            }
        };

        Self {
            tt: Arc::new(Mutex::new(HashMap::new())),
            stop: Arc::new(AtomicBool::new(false)),
            handle: None,
            difficulty: Arc::new(Mutex::new(5)), // Default to Expert level
            ml_evaluator: Arc::new(Mutex::new(ml_eval)),
            use_ml: Arc::new(Mutex::new(false)), // ML disabled by default (uses random weights, very slow)
            // XBoard defaults
            force_mode: false,
            post_mode: false,
            engine_color: None,
            xboard_time: 30000,      // 5 minutes default
            xboard_otim: 30000,
            time_per_move: None,
            depth_limit: None,
            level_moves: 0,          // 0 = game in X
            level_base: 30000,       // 5 minutes in centiseconds
            level_inc: 0,
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
    /// Evaluate a position using ML (if enabled and available) or classical evaluation
    fn eval(&self, b: &Board) -> i32 {
        // Check if ML evaluation is enabled
        let use_ml = *self.searcher.use_ml.lock().unwrap();

        if use_ml {
            // Try ML evaluation
            if let Some(ref ml_eval) = *self.searcher.ml_evaluator.lock().unwrap() {
                match ml_eval.evaluate(b) {
                    Ok(score) => return score,
                    Err(_) => {
                        // Fall through to classical evaluation
                    }
                }
            }
        }

        // Fallback to classical evaluation
        eval_classical(b)
    }

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

                // Output thinking in post mode (XBoard protocol)
                if self.searcher.post_mode {
                    let elapsed_cs = self.start.elapsed().as_millis() as i32 / 10;
                    if let Some(m) = best {
                        println!("{} {} {} {} {}", d, sc, elapsed_cs, self.nodes, move_to_str(m));
                    }
                }
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
        let static_eval = self.eval(b);
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
        let stand = self.eval(b);
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
    if let Some(pv) = pv_move
        && pv == *m {
            return 1_000_000;
        }
    if killers.contains(&Some(*m)) {
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
        if count > 0
            && (pawns & ADJACENT_FILES[file]) == 0 {
                penalty += ISOLATED_PAWN_PENALTY;
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

/// Calculate time allocation for XBoard mode based on current time controls
fn calculate_xboard_time(s: &Searcher, _b: &Board) -> i32 {
    // If fixed time per move is set, use that
    if let Some(time_per_move) = s.time_per_move {
        return time_per_move;
    }

    // Calculate time based on remaining time
    // Use a simple allocation: 1/40th of remaining time + increment
    // This is a conservative allocation suitable for most time controls
    let base_time = s.xboard_time / 40;
    base_time + s.level_inc
}

fn handle_line(line: &str, b: &mut Board, s: &mut Searcher, xboard_mode: &mut bool) {
    let line = line.trim();
    if line.is_empty() {
        return;
    }

    // UCI protocol
    if line == "uci" {
        println!("id name rce-ml");
        println!("id author openai + neural network");
        println!("option name Skill Level type spin default 5 min 1 max 5");
        println!("option name Use ML Evaluation type check default false");
        println!("option name ML Model Path type string default <empty>");

        // Print ML status
        if let Some(ref ml_eval) = *s.ml_evaluator.lock().unwrap() {
            eprintln!("# ML Status: {}", ml_eval.device_info());
        }

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
        let use_ml = *s.use_ml.lock().unwrap();
        let time_ms = time_for_move(b.stm, &params, use_ml);
        let depth = get_depth_for_difficulty(difficulty, params.depth, use_ml, time_ms);
        let mut b2 = b.clone();
        let tt = s.tt.clone();
        let stop = s.stop.clone();
        let diff = s.difficulty.clone();
        let ml_eval = s.ml_evaluator.clone();
        let use_ml = s.use_ml.clone();
        s.handle = Some(std::thread::spawn(move || {
            stop.store(false, Ordering::Relaxed);
            let mut si = SearchInstance {
                searcher: Searcher {
                    tt,
                    stop,
                    handle: None,
                    difficulty: diff,
                    ml_evaluator: ml_eval,
                    use_ml,
                    force_mode: false,
                    post_mode: false,
                    engine_color: None,
                    xboard_time: 0,
                    xboard_otim: 0,
                    time_per_move: None,
                    depth_limit: None,
                    level_moves: 0,
                    level_base: 0,
                    level_inc: 0,
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
    if line.starts_with("protover") {
        // XBoard protocol version 2 feature negotiation
        println!("feature myname=\"rce-ml\" variants=\"normal\" done=0");
        println!("feature setboard=1 usermove=0 time=1 draw=1 sigint=0 sigterm=0");
        println!("feature reuse=1 analyze=0 colors=0 names=0");
        println!("feature ping=1 playother=1 san=0 debug=1");
        println!("feature memory=0 smp=0 egt=\"\"");
        println!("feature option=\"Skill Level -spin 5 1 5\"");
        println!("feature option=\"Use ML Evaluation -check 0\"");
        println!("feature done=1");
        return;
    }
    if line == "new" {
        *b = Board::startpos();
        s.tt.lock().unwrap().clear();
        s.force_mode = false;
        s.engine_color = Some(Side::Black); // Engine plays black after "new"
        return;
    }
    if line.starts_with("setboard ") {
        xboard_set_position(line, b);
        return;
    }
    if line.starts_with("go") || line == "go" {
        s.force_mode = false;
        s.engine_color = Some(b.stm);

        let difficulty = *s.difficulty.lock().unwrap();
        let use_ml = *s.use_ml.lock().unwrap();

        let time_cs = if *xboard_mode {
            calculate_xboard_time(s, b)
        } else {
            5000 // 5 seconds default for non-xboard mode
        };

        let time_ms = (time_cs * 10) as u64;
        let depth = if let Some(d) = s.depth_limit {
            // Respect depth limit but apply adaptive capping
            get_depth_for_difficulty(difficulty, Some(d), use_ml, time_ms)
        } else {
            get_depth_for_difficulty(difficulty, None, use_ml, time_ms)
        };

        let mut b2 = b.clone();
        let tt = s.tt.clone();
        let stop = s.stop.clone();
        let diff = s.difficulty.clone();
        let ml_eval = s.ml_evaluator.clone();
        let use_ml = s.use_ml.clone();
        let post_mode = s.post_mode;

        s.handle = Some(std::thread::spawn(move || {
            stop.store(false, Ordering::Relaxed);
            let mut si = SearchInstance {
                searcher: Searcher {
                    tt,
                    stop,
                    handle: None,
                    difficulty: diff,
                    ml_evaluator: ml_eval,
                    use_ml,
                    force_mode: false,
                    post_mode,
                    engine_color: None,
                    xboard_time: 0,
                    xboard_otim: 0,
                    time_per_move: None,
                    depth_limit: None,
                    level_moves: 0,
                    level_base: 0,
                    level_inc: 0,
                },
                nodes: 0,
                start: Instant::now(),
                time_limit: Duration::from_millis((time_cs * 10) as u64),
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

        // In XBoard mode, if not in force mode and it's now the engine's turn, search
        if *xboard_mode && !s.force_mode {
            if let Some(engine_color) = s.engine_color {
                if b.stm == engine_color {
                    // Automatically search and move
                    let difficulty = *s.difficulty.lock().unwrap();
                    let use_ml = *s.use_ml.lock().unwrap();

                    let time_cs = calculate_xboard_time(s, b);
                    let time_ms = (time_cs * 10) as u64;

                    let depth = if let Some(d) = s.depth_limit {
                        get_depth_for_difficulty(difficulty, Some(d), use_ml, time_ms)
                    } else {
                        get_depth_for_difficulty(difficulty, None, use_ml, time_ms)
                    };
                    let mut b2 = b.clone();
                    let tt = s.tt.clone();
                    let stop = s.stop.clone();
                    let diff = s.difficulty.clone();
                    let ml_eval = s.ml_evaluator.clone();
                    let use_ml = s.use_ml.clone();
                    let post_mode = s.post_mode;

                    s.handle = Some(std::thread::spawn(move || {
                        stop.store(false, Ordering::Relaxed);
                        let mut si = SearchInstance {
                            searcher: Searcher {
                                tt,
                                stop,
                                handle: None,
                                difficulty: diff,
                                ml_evaluator: ml_eval,
                                use_ml,
                                force_mode: false,
                                post_mode,
                                engine_color: None,
                                xboard_time: 0,
                                xboard_otim: 0,
                                time_per_move: None,
                                depth_limit: None,
                                level_moves: 0,
                                level_base: 0,
                                level_inc: 0,
                            },
                            nodes: 0,
                            start: Instant::now(),
                            time_limit: Duration::from_millis((time_cs * 10) as u64),
                            killers: [[None; 2]; MAX_PLY],
                            history: [[[0; 64]; 64]; 2],
                        };
                        let (_score, mv) = si.search(&mut b2, depth);
                        if let Some(m) = mv {
                            println!("move {}", move_to_str(m));
                        }
                    }));
                }
            }
        }
        return;
    }
    if line.starts_with("hint") {
        let difficulty = *s.difficulty.lock().unwrap();
        let use_ml_flag = *s.use_ml.lock().unwrap();
        let time_ms = 2000; // 2 seconds for hints
        let depth = get_depth_for_difficulty(difficulty, Some(4), use_ml_flag, time_ms);
        let mut b2 = b.clone();
        let tt = s.tt.clone();
        let stop = s.stop.clone();
        let diff = s.difficulty.clone();
        let ml_eval = s.ml_evaluator.clone();
        let use_ml = s.use_ml.clone();
        s.handle = Some(std::thread::spawn(move || {
            stop.store(false, Ordering::Relaxed);
            let mut si = SearchInstance {
                searcher: Searcher {
                    tt,
                    stop,
                    handle: None,
                    difficulty: diff,
                    ml_evaluator: ml_eval,
                    use_ml,
                    force_mode: false,
                    post_mode: false,
                    engine_color: None,
                    xboard_time: 0,
                    xboard_otim: 0,
                    time_per_move: None,
                    depth_limit: None,
                    level_moves: 0,
                    level_base: 0,
                    level_inc: 0,
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
        s.force_mode = true;
        // Stop any ongoing search
        s.stop.store(true, Ordering::Relaxed);
        if let Some(h) = s.handle.take() {
            let _ = h.join();
        }
        return;
    }
    if line == "random" {
        // Randomness is controlled via difficulty levels
        return;
    }
    if line.starts_with("level ") {
        // Format: level MOVES BASE INC
        // MOVES = moves per time control (0 for game in BASE)
        // BASE = base time in minutes
        // INC = increment in seconds
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 4 {
            if let (Ok(moves), Ok(base), Ok(inc)) = (
                parts[1].parse::<i32>(),
                parts[2].parse::<i32>(),
                parts[3].parse::<i32>(),
            ) {
                s.level_moves = moves;
                s.level_base = base * 60 * 100; // Convert minutes to centiseconds
                s.level_inc = inc * 100; // Convert seconds to centiseconds
                // Reset time to base time
                s.xboard_time = s.level_base;
            }
        }
        return;
    }
    if line.starts_with("st ") {
        // Set fixed time per move in seconds
        if let Some(time_str) = line.split_whitespace().nth(1) {
            if let Ok(seconds) = time_str.parse::<i32>() {
                s.time_per_move = Some(seconds * 100); // Convert to centiseconds
            }
        }
        return;
    }
    if line.starts_with("sd ") {
        // Set depth limit
        if let Some(depth_str) = line.split_whitespace().nth(1) {
            if let Ok(depth) = depth_str.parse::<i32>() {
                s.depth_limit = Some(depth);
            }
        }
        return;
    }
    if line.starts_with("time ") {
        // Set engine's remaining time in centiseconds
        if let Some(time_str) = line.split_whitespace().nth(1) {
            if let Ok(time) = time_str.parse::<i32>() {
                s.xboard_time = time;
            }
        }
        return;
    }
    if line.starts_with("otim ") {
        // Set opponent's remaining time in centiseconds
        if let Some(time_str) = line.split_whitespace().nth(1) {
            if let Ok(time) = time_str.parse::<i32>() {
                s.xboard_otim = time;
            }
        }
        return;
    }
    if line == "remove" {
        if !b.hist.is_empty() {
            b.unmake();
        }
        if !b.hist.is_empty() {
            b.unmake();
        }
        return;
    }
    if line == "undo" {
        if !b.hist.is_empty() {
            b.unmake();
        }
        return;
    }

    if line.starts_with("ping ") {
        // Respond with pong and the same number
        if let Some(num) = line.split_whitespace().nth(1) {
            println!("pong {}", num);
        }
        return;
    }
    if line == "?" {
        // Move now - interrupt search and return best move so far
        s.stop.store(true, Ordering::Relaxed);
        return;
    }
    if line == "playother" {
        // Switch sides - engine plays the opposite color
        s.engine_color = Some(if b.stm == Side::White { Side::Black } else { Side::White });
        s.force_mode = false;
        return;
    }
    if line == "post" {
        s.post_mode = true;
        return;
    }
    if line == "nopost" {
        s.post_mode = false;
        return;
    }
    if line == "hard" {
        // Enable pondering (not implemented)
        return;
    }
    if line == "easy" {
        // Disable pondering (not implemented)
        return;
    }
    if line.starts_with("result ") {
        // Game ended - just acknowledge
        return;
    }
    if line == "computer" {
        // Opponent is a computer
        return;
    }
    if line.starts_with("name ") {
        // Opponent's name
        return;
    }
    if line.starts_with("rating ") {
        // Player ratings
        return;
    }
    if line == "draw" {
        // Draw offer - decline for now
        return;
    }
    if line.starts_with("accepted ") || line.starts_with("rejected ") {
        // Feature negotiation responses
        return;
    }
    if line == "quit" {
        std::process::exit(0);
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
        } else if name.contains("use ml") || name.contains("useml") {
            // Handle "Use ML Evaluation" option
            let value_str = parts[v_idx].to_lowercase();
            let enabled = value_str == "true" || value_str == "1";
            *s.use_ml.lock().unwrap() = enabled;
            println!("info string ML evaluation {}", if enabled { "enabled" } else { "disabled" });
        } else if name.contains("ml model") || name.contains("mlmodel") {
            // Handle "ML Model Path" option
            let path = parts[v_idx..].join(" ");
            if path != "<empty>" && !path.is_empty() {
                println!("info string Loading ML model from: {}", path);
                // TODO: Implement model loading
                // For now, just acknowledge the path
            }
        }
    }
}

fn get_depth_for_difficulty(difficulty: u8, requested_depth: Option<i32>, use_ml: bool, time_ms: u64) -> i32 {
    // Calculate adaptive max depth based on time and ML usage
    let adaptive_max_depth = if use_ml {
        // ML is very slow, limit depth based on time
        if time_ms < 1000 {
            3  // < 1 second: very shallow
        } else if time_ms < 5000 {
            5  // < 5 seconds: shallow
        } else if time_ms < 15000 {
            7  // < 15 seconds: moderate
        } else if time_ms < 30000 {
            10 // < 30 seconds: deep
        } else {
            12 // >= 30 seconds: very deep (but still limited for ML)
        }
    } else {
        // Classical evaluation is fast, can search deeper
        if time_ms < 1000 {
            6  // < 1 second
        } else if time_ms < 5000 {
            10 // < 5 seconds
        } else if time_ms < 15000 {
            15 // < 15 seconds
        } else {
            100 // >= 15 seconds: search until time runs out
        }
    };

    // If depth is explicitly requested, respect it but cap at adaptive max
    if let Some(depth) = requested_depth {
        return depth.min(adaptive_max_depth);
    }

    // Otherwise use difficulty-based depth, capped at adaptive max
    let difficulty_depth = match difficulty {
        1 => 2,  // Beginner
        2 => 3,  // Easy
        3 => 5,  // Medium
        4 => 7,  // Hard
        _ => adaptive_max_depth, // Expert: use adaptive max
    };

    difficulty_depth.min(adaptive_max_depth)
}

fn time_for_move(stm: Side, params: &GoParams, use_ml: bool) -> u64 {
    if let Some(mt) = params.movetime {
        // Apply hard limit even when movetime is specified
        return mt.min(MAX_MOVE_TIME_MS);
    }
    let (time, inc) = if stm == Side::White {
        (params.wtime, params.winc)
    } else {
        (params.btime, params.binc)
    };
    let moves_to_go = params.movestogo.unwrap_or(40).max(1);
    let mut allocated = match time {
        Some(t) => {
            let base = t / moves_to_go;
            let bonus = inc.unwrap_or(0) / 2;
            let slice = base + bonus;
            let max_allowed = t.saturating_sub(TIME_SAFETY_MARGIN_MS).max(MIN_MOVE_TIME_MS);
            slice.max(MIN_MOVE_TIME_MS).min(max_allowed)
        }
        None => 5000, // Default 5 seconds
    };

    // Reduce time allocation when ML is enabled (ML is much slower)
    if use_ml {
        allocated = ((allocated as f64) * ML_TIME_MULTIPLIER) as u64;
        allocated = allocated.max(MIN_MOVE_TIME_MS);
    }

    // Apply hard maximum time limit
    allocated.min(MAX_MOVE_TIME_MS)
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
