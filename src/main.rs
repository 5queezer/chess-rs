use std::collections::HashMap;
use std::io::{self, Read};
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
