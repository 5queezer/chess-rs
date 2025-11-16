//! Test harness to verify chess engine movement and UCI protocol compliance

mod common;

use common::{is_legal_white_opening_square, is_valid_uci_move, EngineProcess};
use std::time::Duration;

#[test]
fn test_uci_handshake() {
    let mut engine = EngineProcess::new();
    std::thread::sleep(Duration::from_millis(200));
    engine.drain_output(Duration::from_millis(500));

    engine.send("uci");
    let (output, found) = engine.read_until("uciok", Duration::from_secs(5));
    assert!(found, "UCI handshake failed: {:?}", output);

    engine.send("isready");
    let (output, found) = engine.read_until("readyok", Duration::from_secs(5));
    assert!(found, "Engine not ready: {:?}", output);
}

#[test]
fn test_move_generation() {
    let mut engine = EngineProcess::new();
    std::thread::sleep(Duration::from_millis(300));
    engine.drain_output(Duration::from_millis(500));

    assert!(engine.init_uci(), "UCI initialization failed");

    engine.send("position startpos");
    engine.send("go depth 4");

    let (lines, found) = engine.read_until("bestmove", Duration::from_secs(30));
    assert!(found, "No bestmove returned: {:?}", lines);

    let bestmove_line = lines
        .iter()
        .rev()
        .find(|l| l.starts_with("bestmove"))
        .expect("No bestmove line found");

    let parts: Vec<&str> = bestmove_line.split_whitespace().collect();
    assert!(parts.len() >= 2, "Invalid bestmove format: {}", bestmove_line);

    let mv = parts[1];
    assert!(is_valid_uci_move(mv), "Invalid UCI move format: {}", mv);

    let start_square = &mv[0..2];
    assert!(
        is_legal_white_opening_square(start_square),
        "Move starts from illegal square: {}",
        start_square
    );
}

#[test]
fn test_multiple_moves() {
    let mut engine = EngineProcess::new();
    std::thread::sleep(Duration::from_millis(300));
    engine.drain_output(Duration::from_millis(500));

    assert!(engine.init_uci(), "UCI initialization failed");

    let mut moves = Vec::new();
    let opponent_moves = ["e2e4", "d2d4", "g1f3", "b1c3", "f1c4"];

    for (i, opp_move) in opponent_moves.iter().enumerate() {
        let position = if i == 0 {
            "position startpos".to_string()
        } else {
            format!("position startpos moves {}", moves.join(" "))
        };

        let mv = engine.get_move_uci(&position, "go depth 3", Duration::from_secs(15));
        assert!(mv.is_some(), "No move returned at position {}", i + 1);

        let engine_move = mv.unwrap();
        assert!(
            is_valid_uci_move(&engine_move),
            "Invalid move at position {}: {}",
            i + 1,
            engine_move
        );

        moves.push(engine_move);

        if i < opponent_moves.len() - 1 {
            moves.push(opp_move.to_string());
        }
    }

    assert_eq!(
        moves.len(),
        opponent_moves.len() * 2 - 1,
        "Unexpected number of moves"
    );
}

#[test]
fn test_xboard_mode() {
    let mut engine = EngineProcess::new();
    std::thread::sleep(Duration::from_millis(300));
    engine.drain_output(Duration::from_millis(500));

    engine.send("xboard");
    engine.send("protover 2");

    let (output, found) = engine.read_until("done=1", Duration::from_secs(5));
    assert!(found, "XBoard protover handshake failed: {:?}", output);

    engine.send("new");
    std::thread::sleep(Duration::from_millis(200));

    engine.send("go");
    let (lines, _) = engine.read_until("move ", Duration::from_secs(20));

    let move_line = lines.iter().find(|l| l.starts_with("move "));
    assert!(
        move_line.is_some(),
        "No move in XBoard mode. Output: {:?}",
        lines
    );
}

#[test]
fn test_search_info() {
    let mut engine = EngineProcess::new();
    std::thread::sleep(Duration::from_millis(300));
    engine.drain_output(Duration::from_millis(500));

    assert!(engine.init_uci(), "UCI initialization failed");

    engine.send("position startpos");
    engine.send("go depth 5");

    let (lines, found) = engine.read_until("bestmove", Duration::from_secs(30));
    assert!(found, "No bestmove returned");

    let info_lines: Vec<_> = lines.iter().filter(|l| l.starts_with("info")).collect();
    assert!(
        !info_lines.is_empty(),
        "No info lines output during search"
    );

    // Check that info lines contain depth information
    let has_depth_info = info_lines.iter().any(|l| l.contains("depth"));
    assert!(has_depth_info, "Info lines don't contain depth information");
}
