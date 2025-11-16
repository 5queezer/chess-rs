//! Common test utilities for chess engine integration tests

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

/// Chess engine process wrapper with threaded I/O
pub struct EngineProcess {
    pub child: Child,
    output_rx: Receiver<String>,
    _reader_thread: thread::JoinHandle<()>,
}

impl EngineProcess {
    /// Start a new chess engine process
    pub fn new() -> Self {
        Self::with_path("./target/release/chess")
    }

    /// Start a chess engine process with custom path
    pub fn with_path(path: &str) -> Self {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to start chess engine");

        let stdout = child.stdout.take().expect("Failed to get stdout");
        let (tx, rx): (Sender<String>, Receiver<String>) = mpsc::channel();

        let reader_thread = thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                if let Ok(line) = line {
                    if tx.send(line).is_err() {
                        break;
                    }
                }
            }
        });

        EngineProcess {
            child,
            output_rx: rx,
            _reader_thread: reader_thread,
        }
    }

    /// Send a command to the engine
    pub fn send(&mut self, cmd: &str) {
        if let Some(ref mut stdin) = self.child.stdin {
            writeln!(stdin, "{}", cmd).expect("Failed to write to engine");
            stdin.flush().expect("Failed to flush stdin");
        }
    }

    /// Read lines until a pattern is found or timeout
    pub fn read_until(&mut self, pattern: &str, timeout: Duration) -> (Vec<String>, bool) {
        let start = Instant::now();
        let mut lines = Vec::new();

        while start.elapsed() < timeout {
            match self.output_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(line) => {
                    let found = line.contains(pattern);
                    lines.push(line);
                    if found {
                        return (lines, true);
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        (lines, false)
    }

    /// Drain any pending output
    pub fn drain_output(&mut self, timeout: Duration) -> Vec<String> {
        let start = Instant::now();
        let mut lines = Vec::new();

        while start.elapsed() < timeout {
            match self.output_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(line) => lines.push(line),
                Err(_) => break,
            }
        }
        lines
    }

    /// Initialize engine in UCI mode
    pub fn init_uci(&mut self) -> bool {
        self.send("uci");
        let (_, found) = self.read_until("uciok", Duration::from_secs(5));
        if !found {
            return false;
        }

        self.send("isready");
        let (_, found) = self.read_until("readyok", Duration::from_secs(5));
        found
    }

    /// Initialize engine in XBoard mode
    pub fn init_xboard(&mut self) -> bool {
        self.send("xboard");
        thread::sleep(Duration::from_millis(100));

        self.send("protover 2");
        let (_, found) = self.read_until("done=1", Duration::from_secs(5));
        if !found {
            return false;
        }

        self.send("new");
        thread::sleep(Duration::from_millis(100));
        true
    }

    /// Get a move from the engine in UCI mode
    pub fn get_move_uci(&mut self, position: &str, search_params: &str, timeout: Duration) -> Option<String> {
        self.send(position);
        self.send(search_params);

        let (lines, found) = self.read_until("bestmove", timeout);
        if !found {
            return None;
        }

        for line in lines.iter().rev() {
            if line.starts_with("bestmove") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    return Some(parts[1].to_string());
                }
            }
        }
        None
    }

    /// Get a move from the engine in XBoard mode
    pub fn get_move_xboard(&mut self, timeout: Duration) -> Option<String> {
        self.send("go");
        let (lines, _) = self.read_until("move ", timeout);

        for line in lines {
            if line.starts_with("move ") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    return Some(parts[1].to_string());
                }
            }
        }
        None
    }

    /// Terminate the engine process
    pub fn terminate(&mut self) {
        self.send("quit");
        thread::sleep(Duration::from_millis(100));
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for EngineProcess {
    fn drop(&mut self) {
        self.terminate();
    }
}

/// Parse a bestmove line and extract the move
pub fn parse_bestmove(line: &str) -> Option<String> {
    if line.starts_with("bestmove") {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            return Some(parts[1].to_string());
        }
    }
    None
}

/// Check if a move is valid UCI format (e.g., e2e4, a7a8q)
pub fn is_valid_uci_move(mv: &str) -> bool {
    if mv.len() < 4 || mv.len() > 5 {
        return false;
    }

    let bytes = mv.as_bytes();
    let file_valid = |f: u8| f >= b'a' && f <= b'h';
    let rank_valid = |r: u8| r >= b'1' && r <= b'8';

    file_valid(bytes[0])
        && rank_valid(bytes[1])
        && file_valid(bytes[2])
        && rank_valid(bytes[3])
        && (mv.len() == 4 || matches!(bytes[4], b'q' | b'r' | b'b' | b'n'))
}

/// Legal starting squares for white's first move
pub fn is_legal_white_opening_square(square: &str) -> bool {
    matches!(
        square,
        "a2" | "b2" | "c2" | "d2" | "e2" | "f2" | "g2" | "h2" | "b1" | "g1"
    )
}

/// Get the path to the chess engine binary
pub fn engine_path() -> String {
    std::env::var("CHESS_ENGINE_PATH").unwrap_or_else(|_| "./target/release/chess".to_string())
}
