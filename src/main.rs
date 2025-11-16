use std::collections::HashMap;
use std::io::{self, BufRead, Write};
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

const PIECE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 0];

static PIECE_SQUARE_TABLES: [[i32; 64]; 6] = [
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
const ROOK_OPEN_FILE_BONUS: i32 = 25;
const ROOK_SEMI_OPEN_FILE_BONUS: i32 = 12;
const DOUBLED_PAWN_PENALTY: i32 = 12;
const ISOLATED_PAWN_PENALTY: i32 = 15;
const MAX_MOVE_TIME_MS: u64 = 60000;
const MIN_MOVE_TIME_MS: u64 = 50;
const TIME_SAFETY_MARGIN_MS: u64 = 100;
const ML_TIME_MULTIPLIER: f64 = 0.3;
const DEFAULT_SEARCH_TIME_MS: u64 = 5000;
const MATE_SCORE: i32 = 29000;
const HOPELESS_THRESHOLD: i32 = -500;
const HOPELESS_TIMEOUT_SECS: u64 = 60;

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

type HistoryTable = [[[i32; 64]; 64]; 2];

#[derive(Clone)]
struct TranspositionEntry {
    depth: i32,
    score: i32,
    flag: u8,
    best_move: Option<Move>,
}

struct Engine {
    transposition_table: Arc<Mutex<HashMap<u64, TranspositionEntry>>>,
    stop_flag: Arc<AtomicBool>,
    search_thread: Option<JoinHandle<()>>,
    difficulty: Arc<Mutex<u8>>,
    ml_evaluator: Arc<Mutex<Option<MLEvaluator>>>,
    use_ml: Arc<Mutex<bool>>,
    force_mode: bool,
    post_mode: bool,
    engine_color: Option<Side>,
    remaining_time_cs: i32,
    opponent_time_cs: i32,
    time_per_move_cs: Option<i32>,
    depth_limit: Option<i32>,
    moves_per_session: i32,
    base_time_cs: i32,
    increment_cs: i32,
}

impl Engine {
    fn new() -> Self {
        let ml_eval = MLEvaluator::new().ok();
        if let Some(ref evaluator) = ml_eval {
            evaluator.print_status();
        }

        Self {
            transposition_table: Arc::new(Mutex::new(HashMap::new())),
            stop_flag: Arc::new(AtomicBool::new(false)),
            search_thread: None,
            difficulty: Arc::new(Mutex::new(5)),
            ml_evaluator: Arc::new(Mutex::new(ml_eval)),
            use_ml: Arc::new(Mutex::new(false)),
            force_mode: false,
            post_mode: false,
            engine_color: None,
            remaining_time_cs: 30000,
            opponent_time_cs: 30000,
            time_per_move_cs: None,
            depth_limit: None,
            moves_per_session: 0,
            base_time_cs: 30000,
            increment_cs: 0,
        }
    }
}

struct SearchContext {
    engine: Engine,
    nodes_searched: u64,
    start_time: Instant,
    time_limit: Duration,
    killer_moves: [[Option<Move>; 2]; MAX_PLY],
    history_table: HistoryTable,
}

impl SearchContext {
    fn evaluate_position(&self, board: &Board) -> i32 {
        if *self.engine.use_ml.lock().unwrap() {
            if let Some(ref ml_eval) = *self.engine.ml_evaluator.lock().unwrap() {
                if let Ok(score) = ml_eval.evaluate(board) {
                    return score;
                }
            }
        }
        evaluate_classical(board)
    }

    fn find_best_move(&mut self, board: &mut Board, max_depth: i32) -> (i32, Option<Move>) {
        if board.is_draw() {
            let mut moves = Vec::new();
            board.gen_moves(&mut moves);
            return if !moves.is_empty() {
                (0, Some(moves[0]))
            } else {
                (0, None)
            };
        }

        let mut best_move = None;
        let mut best_score = 0;
        let mut previous_score = 0;
        let mut hopeless_since: Option<Instant> = None;

        'depth_iteration: for depth in 1..=max_depth {
            if self.is_time_exceeded() {
                break;
            }

            let (score, mv) = self.search_with_aspiration(board, depth, previous_score);

            if self.engine.stop_flag.load(Ordering::Relaxed) {
                break 'depth_iteration;
            }

            best_score = score;
            if mv.is_some() {
                best_move = mv;
            }
            previous_score = score;

            if self.engine.post_mode {
                self.print_thinking_output(depth, score, best_move);
            }

            if self.should_offer_draw(score, &mut hopeless_since) {
                break 'depth_iteration;
            }

            if self.is_time_exceeded() {
                break;
            }
        }

        let difficulty = *self.engine.difficulty.lock().unwrap();
        if difficulty < 3 && best_move.is_some() {
            best_move = self.apply_randomization(board, best_move, difficulty);
        }

        (best_score, best_move)
    }

    fn search_with_aspiration(&mut self, board: &mut Board, depth: i32, prev_score: i32) -> (i32, Option<Move>) {
        let mut window = ASPIRATION_WINDOW;
        let mut alpha = -30000;
        let mut beta = 30000;
        let use_aspiration = depth > 1;

        if use_aspiration {
            alpha = (prev_score - window).max(-30000);
            beta = (prev_score + window).min(30000);
        }

        loop {
            let (score, best_move) = self.alpha_beta_search(board, depth, alpha, beta, 0);

            if self.engine.stop_flag.load(Ordering::Relaxed) {
                return (score, best_move);
            }

            if use_aspiration && (score <= alpha || score >= beta) {
                window *= 2;
                alpha = (prev_score - window).max(-30000);
                beta = (prev_score + window).min(30000);
                if alpha <= -30000 && beta >= 30000 {
                    continue;
                }
                continue;
            }

            return (score, best_move);
        }
    }

    fn print_thinking_output(&self, depth: i32, score: i32, best_move: Option<Move>) {
        let elapsed_cs = self.start_time.elapsed().as_millis() as i32 / 10;
        if let Some(m) = best_move {
            println!("{} {} {} {} {}", depth, score, elapsed_cs, self.nodes_searched, move_to_str(m));
        }
    }

    fn should_offer_draw(&self, score: i32, hopeless_since: &mut Option<Instant>) -> bool {
        if score < HOPELESS_THRESHOLD {
            if hopeless_since.is_none() {
                *hopeless_since = Some(Instant::now());
            } else if let Some(start) = *hopeless_since {
                if start.elapsed() >= Duration::from_secs(HOPELESS_TIMEOUT_SECS) {
                    println!("offer draw");
                    return true;
                }
            }
        } else {
            *hopeless_since = None;
        }
        false
    }

    fn apply_randomization(&mut self, board: &mut Board, best: Option<Move>, difficulty: u8) -> Option<Move> {
        let mut rng = rand::thread_rng();
        let (randomization_chance, pool_size) = match difficulty {
            1 => (40, 5),
            2 => (20, 3),
            _ => return best,
        };

        if rng.gen_range(0..100) >= randomization_chance {
            return best;
        }

        let legal_moves = self.get_legal_moves(board);
        if legal_moves.is_empty() {
            return best;
        }

        let pick_from = pool_size.min(legal_moves.len());
        Some(legal_moves[rng.gen_range(0..pick_from)])
    }

    fn get_legal_moves(&self, board: &mut Board) -> Vec<Move> {
        let mut moves = Vec::with_capacity(64);
        board.gen_moves(&mut moves);
        moves.into_iter().filter(|&m| {
            board.make_move(m);
            let is_legal = !board.in_check(board.stm.flip());
            board.unmake();
            is_legal
        }).collect()
    }

    fn is_time_exceeded(&self) -> bool {
        self.start_time.elapsed() >= self.time_limit
    }

    fn probe_transposition_table(&self, key: u64) -> Option<TranspositionEntry> {
        self.engine.transposition_table.lock().unwrap().get(&key).cloned()
    }

    fn store_transposition_entry(&mut self, key: u64, entry: TranspositionEntry) {
        self.engine.transposition_table.lock().unwrap().insert(key, entry);
    }

    fn alpha_beta_search(&mut self, board: &mut Board, depth: i32, mut alpha: i32, beta: i32, ply: usize) -> (i32, Option<Move>) {
        if self.is_time_exceeded() || self.engine.stop_flag.load(Ordering::Relaxed) {
            self.engine.stop_flag.store(true, Ordering::Relaxed);
            return (0, None);
        }

        self.nodes_searched += 1;

        if ply > 0 && board.is_draw() {
            return (0, None);
        }

        let side_to_move = board.stm;
        let mut current_depth = depth;
        let in_check = board.in_check(board.stm);

        if in_check {
            current_depth += 1;
        }

        if current_depth <= 0 {
            return (self.quiescence_search(board, alpha, beta), None);
        }

        let static_eval = self.evaluate_position(board);
        let mut pv_move = None;

        if let Some(tt_entry) = self.probe_transposition_table(board.hash) {
            if tt_entry.depth >= current_depth {
                match tt_entry.flag {
                    0 => return (tt_entry.score, tt_entry.best_move),
                    1 => alpha = alpha.max(tt_entry.score),
                    2 if tt_entry.score < beta => return (tt_entry.score, tt_entry.best_move),
                    _ => {}
                }
                if alpha >= beta {
                    return (tt_entry.score, tt_entry.best_move);
                }
            }
            pv_move = tt_entry.best_move;
        }

        if self.should_try_null_move(board, current_depth, in_check, ply, static_eval, beta, side_to_move) {
            let null_state = board.make_null_move();
            let score = -self.alpha_beta_search(board, current_depth - 3, -beta, -beta + 1, ply + 1).0;
            board.unmake_null_move(null_state);
            if score >= beta {
                return (beta, None);
            }
        }

        let mut moves = Vec::with_capacity(64);
        board.gen_moves(&mut moves);

        if moves.is_empty() {
            return if in_check {
                (-MATE_SCORE + ply as i32, None)
            } else {
                (0, None)
            };
        }

        self.order_moves(board, &mut moves, pv_move, ply, side_to_move);

        let mut best_move = None;
        let mut flag = 2;
        let mut legal_move_count = 0;

        for (index, m) in moves.into_iter().enumerate() {
            if self.should_prune_move(m, in_check, current_depth, static_eval, alpha) {
                continue;
            }

            board.make_move(m);
            if board.in_check(board.stm.flip()) {
                board.unmake();
                continue;
            }

            legal_move_count += 1;
            let gives_check = board.in_check(board.stm);
            let new_depth = self.calculate_new_depth(current_depth, index, m, gives_check);

            let score = self.search_move(board, new_depth, alpha, beta, ply, best_move.is_none());
            board.unmake();

            if score > alpha {
                alpha = score;
                best_move = Some(m);
                flag = 0;

                if self.is_quiet_move(m) {
                    self.update_history_table(side_to_move, m, current_depth);
                }

                if alpha >= beta {
                    if self.is_quiet_move(m) {
                        self.store_killer_move(ply, m);
                    }
                    flag = 1;
                    self.store_transposition_entry(board.hash, TranspositionEntry {
                        depth: current_depth,
                        score: alpha,
                        flag,
                        best_move,
                    });
                    return (alpha, best_move);
                }
            }
        }

        if legal_move_count == 0 {
            return if in_check {
                (-MATE_SCORE + ply as i32, None)
            } else {
                (0, None)
            };
        }

        self.store_transposition_entry(board.hash, TranspositionEntry {
            depth: current_depth,
            score: alpha,
            flag,
            best_move,
        });

        (alpha, best_move)
    }

    fn should_try_null_move(&self, board: &Board, depth: i32, in_check: bool, ply: usize, static_eval: i32, beta: i32, side: Side) -> bool {
        depth >= 3 && !in_check && ply < MAX_PLY - 1 && has_non_king_material(board, side) && static_eval >= beta
    }

    fn should_prune_move(&self, m: Move, in_check: bool, depth: i32, static_eval: i32, alpha: i32) -> bool {
        if in_check || depth > 2 || !self.is_quiet_move(m) {
            return false;
        }
        let margin = 100 * depth + 50;
        static_eval + margin <= alpha
    }

    fn is_quiet_move(&self, m: Move) -> bool {
        (m.flags & FLAG_CAPTURE) == 0 && m.promo == 255
    }

    fn calculate_new_depth(&self, depth: i32, move_index: usize, m: Move, gives_check: bool) -> i32 {
        let mut new_depth = depth - 1;
        if depth >= 3 && move_index >= 4 && self.is_quiet_move(m) && !gives_check {
            new_depth -= 1;
        }
        new_depth.max(0)
    }

    fn search_move(&mut self, board: &mut Board, depth: i32, alpha: i32, beta: i32, ply: usize, is_first: bool) -> i32 {
        if is_first {
            -self.alpha_beta_search(board, depth, -beta, -alpha, ply + 1).0
        } else {
            let score = -self.alpha_beta_search(board, depth, -alpha - 1, -alpha, ply + 1).0;
            if score > alpha && score < beta {
                -self.alpha_beta_search(board, depth, -beta, -alpha, ply + 1).0
            } else {
                score
            }
        }
    }

    fn order_moves(&self, board: &Board, moves: &mut Vec<Move>, pv_move: Option<Move>, ply: usize, side: Side) {
        let killers = if ply < MAX_PLY {
            self.killer_moves[ply]
        } else {
            [None, None]
        };
        let history = &self.history_table;
        moves.sort_by(|lhs, rhs| {
            calculate_move_score(board, rhs, pv_move, &killers, side, history)
                .cmp(&calculate_move_score(board, lhs, pv_move, &killers, side, history))
        });
    }

    fn quiescence_search(&mut self, board: &mut Board, mut alpha: i32, beta: i32) -> i32 {
        if self.is_time_exceeded() || self.engine.stop_flag.load(Ordering::Relaxed) {
            self.engine.stop_flag.store(true, Ordering::Relaxed);
            return 0;
        }

        let stand_pat = self.evaluate_position(board);
        if stand_pat >= beta {
            return beta;
        }

        const DELTA_MARGIN: i32 = 200;
        if stand_pat + DELTA_MARGIN < alpha {
            return alpha;
        }

        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let mut moves = Vec::with_capacity(32);
        board.gen_moves(&mut moves);

        for m in moves {
            board.make_move(m);
            if board.in_check(board.stm.flip()) {
                board.unmake();
                continue;
            }

            let is_tactical = (m.flags & FLAG_CAPTURE) != 0 || m.promo != 255 || board.in_check(board.stm);
            if !is_tactical {
                board.unmake();
                continue;
            }

            let score = -self.quiescence_search(board, -beta, -alpha);
            board.unmake();

            if score >= beta {
                return beta;
            }
            if score > alpha {
                alpha = score;
            }
        }

        alpha
    }

    fn store_killer_move(&mut self, ply: usize, m: Move) {
        if ply >= MAX_PLY {
            return;
        }
        if self.killer_moves[ply][0] != Some(m) {
            self.killer_moves[ply][1] = self.killer_moves[ply][0];
            self.killer_moves[ply][0] = Some(m);
        }
    }

    fn update_history_table(&mut self, side: Side, m: Move, depth: i32) {
        if depth <= 0 {
            return;
        }
        let bonus = depth * depth;
        let entry = &mut self.history_table[side as usize][m.from as usize][m.to as usize];
        *entry = (*entry + bonus).min(HISTORY_MAX);
    }
}

fn evaluate_classical(board: &Board) -> i32 {
    let mut score = 0;

    for color in 0..2 {
        for piece in 0..6 {
            let mut bitboard = board.bb_piece[color][piece];
            while bitboard != 0 {
                let square = pop_lsb(&mut bitboard);
                let piece_score = PIECE_VALUES[piece] + piece_square_value(piece, if color == 0 { square } else { 63 - square });
                score += if color == 0 { piece_score } else { -piece_score };
            }
        }
    }

    let white = Side::White as usize;
    let black = Side::Black as usize;

    if board.bb_piece[white][BISHOP].count_ones() >= 2 {
        score += BISHOP_PAIR_BONUS;
    }
    if board.bb_piece[black][BISHOP].count_ones() >= 2 {
        score -= BISHOP_PAIR_BONUS;
    }

    let white_pawns = board.bb_piece[white][PAWN];
    let black_pawns = board.bb_piece[black][PAWN];

    score += calculate_rook_file_bonus(board.bb_piece[white][ROOK], white_pawns, black_pawns);
    score -= calculate_rook_file_bonus(board.bb_piece[black][ROOK], black_pawns, white_pawns);
    score -= calculate_pawn_structure_penalty(white_pawns);
    score += calculate_pawn_structure_penalty(black_pawns);

    score
}

#[inline]
fn piece_square_value(piece: usize, square: usize) -> i32 {
    PIECE_SQUARE_TABLES[piece][square]
}

fn calculate_move_score(board: &Board, m: &Move, pv_move: Option<Move>, killers: &[Option<Move>; 2], side: Side, history: &HistoryTable) -> i32 {
    if pv_move == Some(*m) {
        return 1_000_000;
    }
    if killers.contains(&Some(*m)) {
        return 900_000;
    }
    if (m.flags & FLAG_CAPTURE) != 0 {
        return 800_000 + calculate_mvv_lva(board, m, side);
    }
    if m.promo != 255 {
        return 700_000;
    }
    500_000 + history[side as usize][m.from as usize][m.to as usize]
}

fn calculate_mvv_lva(board: &Board, m: &Move, side: Side) -> i32 {
    let victim_value = board.piece_at(side.flip(), m.to as usize).map(|p| PIECE_VALUES[p]).unwrap_or(0);
    let attacker_value = board.piece_at(side, m.from as usize).map(|p| PIECE_VALUES[p]).unwrap_or(0);
    victim_value * 10 - attacker_value
}

fn has_non_king_material(board: &Board, side: Side) -> bool {
    let mut occupancy = board.bb_side[side as usize];
    occupancy &= !board.bb_piece[side as usize][KING];
    occupancy != 0
}

fn calculate_rook_file_bonus(rooks: u64, own_pawns: u64, opponent_pawns: u64) -> i32 {
    let mut bonus = 0;
    for file in 0..8 {
        let mask = FILE_MASKS[file];
        if rooks & mask == 0 {
            continue;
        }
        let has_own_pawn = own_pawns & mask != 0;
        let has_opponent_pawn = opponent_pawns & mask != 0;
        if !has_own_pawn && !has_opponent_pawn {
            bonus += ROOK_OPEN_FILE_BONUS;
        } else if !has_own_pawn && has_opponent_pawn {
            bonus += ROOK_SEMI_OPEN_FILE_BONUS;
        }
    }
    bonus
}

fn calculate_pawn_structure_penalty(pawns: u64) -> i32 {
    let mut penalty = 0;
    for file in 0..8 {
        let mask = FILE_MASKS[file];
        let pawn_count = (pawns & mask).count_ones() as i32;
        if pawn_count > 1 {
            penalty += DOUBLED_PAWN_PENALTY * (pawn_count - 1);
        }
        if pawn_count > 0 && (pawns & ADJACENT_FILES[file]) == 0 {
            penalty += ISOLATED_PAWN_PENALTY;
        }
    }
    penalty
}

fn main() {
    init_tables();
    init_zobrist();

    let _ = io::stdout().flush();

    let stdin = io::stdin();
    let mut board = Board::startpos();
    let mut engine = Engine::new();
    let mut xboard_mode = false;

    for line in stdin.lock().lines().flatten() {
        process_command(line.trim(), &mut board, &mut engine, &mut xboard_mode);
    }

    if let Some(handle) = engine.search_thread.take() {
        let _ = handle.join();
    }
}

fn calculate_xboard_time_allocation(engine: &Engine) -> i32 {
    if let Some(time_per_move) = engine.time_per_move_cs {
        return time_per_move;
    }
    let base_allocation = engine.remaining_time_cs / 40;
    base_allocation + engine.increment_cs
}

fn process_command(line: &str, board: &mut Board, engine: &mut Engine, xboard_mode: &mut bool) {
    let line = line.trim();
    if line.is_empty() {
        return;
    }

    match line {
        "uci" => handle_uci_command(engine),
        "isready" => handle_isready_command(),
        "ucinewgame" => handle_ucinewgame_command(board, engine),
        "stop" => handle_stop_command(engine),
        "xboard" => *xboard_mode = true,
        "new" => handle_new_game_command(board, engine),
        "force" => handle_force_command(engine),
        "random" => {}
        "quit" => std::process::exit(0),
        "?" => engine.stop_flag.store(true, Ordering::Relaxed),
        "playother" => handle_playother_command(board, engine),
        "post" => engine.post_mode = true,
        "nopost" => engine.post_mode = false,
        "hard" | "easy" | "computer" | "draw" => {}
        "remove" => handle_remove_command(board),
        "undo" => handle_undo_command(board),
        _ => process_parameterized_command(line, board, engine, xboard_mode),
    }
}

fn handle_uci_command(engine: &Engine) {
    println!("id name rce-ml");
    println!("id author openai + neural network");
    println!("option name Skill Level type spin default 5 min 1 max 5");
    println!("option name Use ML Evaluation type check default false");
    println!("option name ML Model Path type string default <empty>");
    println!("uciok");
    let _ = io::stdout().flush();
}

fn handle_isready_command() {
    println!("readyok");
    let _ = io::stdout().flush();
}

fn handle_ucinewgame_command(board: &mut Board, engine: &mut Engine) {
    *board = Board::startpos();
    engine.transposition_table.lock().unwrap().clear();
    println!("readyok");
    let _ = io::stdout().flush();
}

fn handle_stop_command(engine: &mut Engine) {
    engine.stop_flag.store(true, Ordering::Relaxed);
    if let Some(handle) = engine.search_thread.take() {
        handle.join().unwrap();
    }
}

fn handle_new_game_command(board: &mut Board, engine: &mut Engine) {
    *board = Board::startpos();
    engine.transposition_table.lock().unwrap().clear();
    engine.force_mode = false;
    engine.engine_color = Some(Side::Black);
}

fn handle_force_command(engine: &mut Engine) {
    engine.force_mode = true;
    engine.stop_flag.store(true, Ordering::Relaxed);
    if let Some(handle) = engine.search_thread.take() {
        let _ = handle.join();
    }
}

fn handle_playother_command(board: &Board, engine: &mut Engine) {
    engine.engine_color = Some(if board.stm == Side::White { Side::Black } else { Side::White });
    engine.force_mode = false;
}

fn handle_remove_command(board: &mut Board) {
    if !board.hist.is_empty() {
        board.unmake();
    }
    if !board.hist.is_empty() {
        board.unmake();
    }
}

fn handle_undo_command(board: &mut Board) {
    if !board.hist.is_empty() {
        board.unmake();
    }
}

fn process_parameterized_command(line: &str, board: &mut Board, engine: &mut Engine, xboard_mode: &mut bool) {
    if line.starts_with("setoption") {
        handle_setoption_command(line, engine);
    } else if line.starts_with("position ") {
        parse_position_command(line, board);
    } else if line.starts_with("go") && !*xboard_mode {
        handle_uci_go_command(line, board, engine);
    } else if line.starts_with("perft ") {
        handle_perft_command(line, board);
    } else if line.starts_with("protover") {
        handle_protover_command();
    } else if line.starts_with("setboard ") {
        parse_xboard_setboard_command(line, board);
    } else if line.starts_with("go") || line == "go" {
        handle_xboard_go_command(board, engine);
    } else if line.starts_with("move ") {
        handle_xboard_move_command(line, board, engine, xboard_mode);
    } else if line.starts_with("hint") {
        handle_hint_command(board, engine);
    } else if line.starts_with("level ") {
        parse_level_command(line, engine);
    } else if line.starts_with("st ") {
        parse_st_command(line, engine);
    } else if line.starts_with("sd ") {
        parse_sd_command(line, engine);
    } else if line.starts_with("time ") {
        parse_time_command(line, engine);
    } else if line.starts_with("otim ") {
        parse_otim_command(line, engine);
    } else if line.starts_with("ping ") {
        handle_ping_command(line);
    } else if line.starts_with("result ") || line.starts_with("name ") || line.starts_with("rating ") || line.starts_with("accepted ") || line.starts_with("rejected ") {
    } else if *xboard_mode && line.len() >= 4 && line.len() <= 5 {
        handle_raw_move_command(line, board, engine);
    }
}

fn handle_setoption_command(cmd: &str, engine: &mut Engine) {
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
                *engine.difficulty.lock().unwrap() = clamped;
            }
        } else if name.contains("use ml") || name.contains("useml") {
            let value_str = parts[v_idx].to_lowercase();
            let enabled = value_str == "true" || value_str == "1";
            *engine.use_ml.lock().unwrap() = enabled;
        }
    }
}

fn calculate_search_depth(difficulty: u8, requested_depth: Option<i32>, use_ml: bool, time_ms: u64) -> i32 {
    let adaptive_max = if use_ml {
        match time_ms {
            0..=999 => 3,
            1000..=4999 => 5,
            5000..=14999 => 7,
            15000..=29999 => 10,
            _ => 12,
        }
    } else {
        match time_ms {
            0..=999 => 6,
            1000..=4999 => 10,
            5000..=14999 => 15,
            _ => 100,
        }
    };

    let difficulty_max = match difficulty {
        1 => 2,
        2 => 4,
        3 => 8,
        4 => 12,
        _ => 100,
    };

    let base_depth = requested_depth.unwrap_or(adaptive_max);
    base_depth.min(difficulty_max).min(adaptive_max)
}

fn calculate_time_for_move(side: Side, params: &GoParams, use_ml: bool) -> u64 {
    if let Some(mt) = params.movetime {
        return mt.min(MAX_MOVE_TIME_MS);
    }

    let (time, inc) = if side == Side::White {
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
        None => DEFAULT_SEARCH_TIME_MS,
    };

    if use_ml {
        allocated = ((allocated as f64) * ML_TIME_MULTIPLIER) as u64;
        allocated = allocated.max(MIN_MOVE_TIME_MS);
    }

    allocated.min(MAX_MOVE_TIME_MS)
}

fn handle_uci_go_command(line: &str, board: &mut Board, engine: &mut Engine) {
    let params = parse_go_params(line);
    let difficulty = *engine.difficulty.lock().unwrap();
    let use_ml = *engine.use_ml.lock().unwrap();
    let time_ms = calculate_time_for_move(board.stm, &params, use_ml);
    let depth = calculate_search_depth(difficulty, params.depth, use_ml, time_ms);

    let board_clone = board.clone();
    let tt = engine.transposition_table.clone();
    let stop = engine.stop_flag.clone();
    let diff = engine.difficulty.clone();
    let ml_eval = engine.ml_evaluator.clone();
    let use_ml_arc = engine.use_ml.clone();

    engine.search_thread = Some(std::thread::spawn(move || {
        stop.store(false, Ordering::Relaxed);
        let mut context = create_search_context(tt, stop, diff, ml_eval, use_ml_arc, time_ms, false);
        let mut board = board_clone;
        let (score, mv) = context.find_best_move(&mut board, depth);

        if let Some(m) = mv {
            print!("info depth {} score cp {} nodes {}\nbestmove {}\n", depth, score, context.nodes_searched, move_to_str(m));
            let _ = io::stdout().flush();
        } else {
            println!("bestmove 0000");
        }
    }));
}

fn handle_xboard_go_command(board: &mut Board, engine: &mut Engine) {
    engine.force_mode = false;
    engine.engine_color = Some(board.stm);
    start_xboard_search(board, engine);
}

fn handle_xboard_move_command(line: &str, board: &mut Board, engine: &mut Engine, xboard_mode: &bool) {
    let move_str = line.strip_prefix("move ").unwrap_or("");
    if let Some(m) = str_to_move(board, move_str) {
        board.make_move(m);

        if *xboard_mode && !engine.force_mode {
            if let Some(engine_color) = engine.engine_color {
                if board.stm == engine_color {
                    start_xboard_search(board, engine);
                }
            }
        }
    }
}

fn handle_raw_move_command(line: &str, board: &mut Board, engine: &mut Engine) {
    if let Some(m) = str_to_move(board, line) {
        board.make_move(m);

        if !engine.force_mode {
            if let Some(engine_color) = engine.engine_color {
                if board.stm == engine_color {
                    start_xboard_search(board, engine);
                }
            }
        }
    }
}

fn start_xboard_search(board: &Board, engine: &mut Engine) {
    let difficulty = *engine.difficulty.lock().unwrap();
    let use_ml = *engine.use_ml.lock().unwrap();
    let time_cs = calculate_xboard_time_allocation(engine);
    let time_ms = (time_cs * 10) as u64;

    let depth = if let Some(d) = engine.depth_limit {
        calculate_search_depth(difficulty, Some(d), use_ml, time_ms)
    } else {
        calculate_search_depth(difficulty, None, use_ml, time_ms)
    };

    let board_clone = board.clone();
    let tt = engine.transposition_table.clone();
    let stop = engine.stop_flag.clone();
    let diff = engine.difficulty.clone();
    let ml_eval = engine.ml_evaluator.clone();
    let use_ml_arc = engine.use_ml.clone();
    let post_mode = engine.post_mode;

    engine.search_thread = Some(std::thread::spawn(move || {
        stop.store(false, Ordering::Relaxed);
        let mut context = create_search_context(tt, stop, diff, ml_eval, use_ml_arc, (time_cs * 10) as u64, post_mode);
        let mut board = board_clone;
        let (_score, mv) = context.find_best_move(&mut board, depth);

        if let Some(m) = mv {
            println!("move {}", move_to_str(m));
            let _ = std::io::stdout().flush();
        }
    }));
}

fn create_search_context(
    tt: Arc<Mutex<HashMap<u64, TranspositionEntry>>>,
    stop: Arc<AtomicBool>,
    difficulty: Arc<Mutex<u8>>,
    ml_evaluator: Arc<Mutex<Option<MLEvaluator>>>,
    use_ml: Arc<Mutex<bool>>,
    time_ms: u64,
    post_mode: bool,
) -> SearchContext {
    SearchContext {
        engine: Engine {
            transposition_table: tt,
            stop_flag: stop,
            search_thread: None,
            difficulty,
            ml_evaluator,
            use_ml,
            force_mode: false,
            post_mode,
            engine_color: None,
            remaining_time_cs: 0,
            opponent_time_cs: 0,
            time_per_move_cs: None,
            depth_limit: None,
            moves_per_session: 0,
            base_time_cs: 0,
            increment_cs: 0,
        },
        nodes_searched: 0,
        start_time: Instant::now(),
        time_limit: Duration::from_millis(time_ms),
        killer_moves: [[None; 2]; MAX_PLY],
        history_table: [[[0; 64]; 64]; 2],
    }
}

fn handle_hint_command(board: &Board, engine: &mut Engine) {
    let difficulty = *engine.difficulty.lock().unwrap();
    let use_ml_flag = *engine.use_ml.lock().unwrap();
    let time_ms = 2000;
    let depth = calculate_search_depth(difficulty, Some(4), use_ml_flag, time_ms);

    let board_clone = board.clone();
    let tt = engine.transposition_table.clone();
    let stop = engine.stop_flag.clone();
    let diff = engine.difficulty.clone();
    let ml_eval = engine.ml_evaluator.clone();
    let use_ml = engine.use_ml.clone();

    engine.search_thread = Some(std::thread::spawn(move || {
        stop.store(false, Ordering::Relaxed);
        let mut context = create_search_context(tt, stop, diff, ml_eval, use_ml, 2000, false);
        let mut board = board_clone;
        let (_score, mv) = context.find_best_move(&mut board, depth);

        if let Some(m) = mv {
            println!("Hint: {}", move_to_str(m));
        }
    }));
}

fn handle_perft_command(line: &str, board: &Board) {
    let depth = line.split_whitespace().nth(1).and_then(|x| x.parse::<u32>().ok()).unwrap_or(3);
    perft::test(board, depth);
}

fn handle_protover_command() {
    println!("feature myname=\"rce-ml\" variants=\"normal\" done=0");
    println!("feature setboard=1 usermove=0 time=1 draw=1 sigint=0 sigterm=0");
    println!("feature reuse=1 analyze=0 colors=0 names=0");
    println!("feature ping=1 playother=1 san=0 debug=1");
    println!("feature memory=0 smp=0 egt=\"\"");
    println!("feature option=\"Skill Level -spin 5 1 5\"");
    println!("feature option=\"Use ML Evaluation -check 0\"");
    println!("feature done=1");
    let _ = io::stdout().flush();
}

fn handle_ping_command(line: &str) {
    if let Some(num) = line.split_whitespace().nth(1) {
        println!("pong {}", num);
        let _ = io::stdout().flush();
    }
}

fn parse_level_command(line: &str, engine: &mut Engine) {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() >= 4 {
        if let (Ok(moves), Ok(base), Ok(inc)) = (parts[1].parse::<i32>(), parts[2].parse::<i32>(), parts[3].parse::<i32>()) {
            engine.moves_per_session = moves;
            engine.base_time_cs = base * 60 * 100;
            engine.increment_cs = inc * 100;
            engine.remaining_time_cs = engine.base_time_cs;
        }
    }
}

fn parse_st_command(line: &str, engine: &mut Engine) {
    if let Some(time_str) = line.split_whitespace().nth(1) {
        if let Ok(seconds) = time_str.parse::<i32>() {
            engine.time_per_move_cs = Some(seconds * 100);
        }
    }
}

fn parse_sd_command(line: &str, engine: &mut Engine) {
    if let Some(depth_str) = line.split_whitespace().nth(1) {
        if let Ok(depth) = depth_str.parse::<i32>() {
            engine.depth_limit = Some(depth);
        }
    }
}

fn parse_time_command(line: &str, engine: &mut Engine) {
    if let Some(time_str) = line.split_whitespace().nth(1) {
        if let Ok(time) = time_str.parse::<i32>() {
            engine.remaining_time_cs = time;
        }
    }
}

fn parse_otim_command(line: &str, engine: &mut Engine) {
    if let Some(time_str) = line.split_whitespace().nth(1) {
        if let Ok(time) = time_str.parse::<i32>() {
            engine.opponent_time_cs = time;
        }
    }
}

fn parse_xboard_setboard_command(cmd: &str, board: &mut Board) {
    let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
    if parts.len() == 2 {
        *board = Board::from_fen(parts[1]);
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

fn parse_go_params(s: &str) -> GoParams {
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

fn parse_position_command(cmd: &str, board: &mut Board) {
    let mut it = cmd.split_whitespace();
    it.next();

    match it.next() {
        Some("startpos") => *board = Board::startpos(),
        Some("fen") => {
            let fen: Vec<&str> = it.clone().take(6).collect();
            *board = Board::from_fen(&fen.join(" "));
            for _ in 0..6 {
                it.next();
            }
        }
        _ => {}
    }

    if let Some("moves") = it.next() {
        for mv in it {
            if let Some(m) = str_to_move(board, mv) {
                board.make_move(m);
            }
        }
    }
}
