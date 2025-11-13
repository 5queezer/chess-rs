use std::collections::HashMap;
use std::io::{self, Read};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

mod board;
mod perft;

use board::*;
use perft::*;

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
}

impl Searcher {
    fn new() -> Self {
        Self {
            tt: Arc::new(Mutex::new(HashMap::new())),
            stop: Arc::new(AtomicBool::new(false)),
            handle: None,
        }
    }
}

struct SearchInstance {
    searcher: Searcher,
    nodes: u64,
    start: Instant,
    time_limit: Duration,
}

impl SearchInstance {
    fn search(&mut self, b: &mut Board, depth: i32) -> (i32, Option<Move>) {
        let mut best = None;
        let mut score = 0;
        for d in 1..=depth {
            let (sc, bm) = self.alpha_beta(b, d, -30000, 30000);
            if self.searcher.stop.load(Ordering::Relaxed) {
                break;
            }
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
    ) -> (i32, Option<Move>) {
        if self.timed_out() || self.searcher.stop.load(Ordering::Relaxed) {
            self.searcher.stop.store(true, Ordering::Relaxed);
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
            if b.in_check(b.stm.flip()) {
                b.unmake();
                continue;
            }
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
        if self.timed_out() || self.searcher.stop.load(Ordering::Relaxed) {
            self.searcher.stop.store(true, Ordering::Relaxed);
            return 0;
        }
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
        return;
    }
    if line.starts_with("position ") {
        set_position(line, b);
        return;
    }
    if line.starts_with("go") {
        let params = parse_go(line);
        let depth = params.depth.unwrap_or(100);
        let time_ms = time_for_move(b.stm, &params);
        let mut b2 = b.clone();
        let tt = s.tt.clone();
        let stop = s.stop.clone();
        s.handle = Some(std::thread::spawn(move || {
            stop.store(false, Ordering::Relaxed);
            let mut si = SearchInstance {
                searcher: Searcher {
                    tt,
                    stop,
                    handle: None,
                },
                nodes: 0,
                start: Instant::now(),
                time_limit: Duration::from_millis(time_ms),
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
        let depth = 6;
        let mut b2 = b.clone();
        let tt = s.tt.clone();
        let stop = s.stop.clone();
        s.handle = Some(std::thread::spawn(move || {
            stop.store(false, Ordering::Relaxed);
            let mut si = SearchInstance {
                searcher: Searcher {
                    tt,
                    stop,
                    handle: None,
                },
                nodes: 0,
                start: Instant::now(),
                time_limit: Duration::from_millis(5000),
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
        let depth = 4;
        let mut b2 = b.clone();
        let tt = s.tt.clone();
        let stop = s.stop.clone();
        s.handle = Some(std::thread::spawn(move || {
            stop.store(false, Ordering::Relaxed);
            let mut si = SearchInstance {
                searcher: Searcher {
                    tt,
                    stop,
                    handle: None,
                },
                nodes: 0,
                start: Instant::now(),
                time_limit: Duration::from_millis(2000),
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

fn time_for_move(stm: Side, params: &GoParams) -> u64 {
    if let Some(mt) = params.movetime {
        return mt;
    }
    let time = if stm == Side::White {
        params.wtime
    } else {
        params.btime
    };
    if let Some(t) = time {
        t / 20
    } else {
        5000
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
    movetime: Option<u64>,
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
            "movetime" => params.movetime = it.next().and_then(|x| x.parse().ok()),
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

