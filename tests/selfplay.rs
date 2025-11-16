//! Self-play test: Have the chess engine play against itself via UCI protocol

mod common;

use common::{is_valid_uci_move, EngineProcess};
use std::time::Duration;

#[test]
fn test_selfplay_basic() {
    let mut white = EngineProcess::new();
    let mut black = EngineProcess::new();

    std::thread::sleep(Duration::from_millis(300));
    white.drain_output(Duration::from_millis(500));
    black.drain_output(Duration::from_millis(500));

    // UCI handshake for both
    assert!(white.init_uci(), "White engine UCI handshake failed");
    assert!(black.init_uci(), "Black engine UCI handshake failed");

    let mut moves = Vec::new();
    let max_moves = 50; // 50 moves each side

    for i in 0..(max_moves * 2) {
        let current_side = if i % 2 == 0 { "White" } else { "Black" };
        let engine = if i % 2 == 0 {
            &mut white
        } else {
            &mut black
        };

        let position = if moves.is_empty() {
            "position startpos".to_string()
        } else {
            format!("position startpos moves {}", moves.join(" "))
        };

        let bestmove = engine.get_move_uci(&position, "go depth 4", Duration::from_secs(30));

        let bestmove = match bestmove {
            Some(m) => m,
            None => {
                panic!("{} failed to return a move after {} half-moves", current_side, moves.len());
            }
        };

        // Check for special cases
        if bestmove == "0000" || bestmove == "(none)" {
            println!("Game Over: {} has no legal moves (checkmate or stalemate)", current_side);
            break;
        }

        assert!(
            is_valid_uci_move(&bestmove),
            "Invalid move from {}: {}",
            current_side,
            bestmove
        );

        moves.push(bestmove);

        // Check for game over conditions (50-move rule)
        if moves.len() >= 100 {
            println!("Game Over: 50-move rule");
            break;
        }

        if moves.len() >= 200 {
            println!("Game Over: Too many moves");
            break;
        }
    }

    assert!(
        moves.len() > 10,
        "Game was too short ({} moves), might indicate issues",
        moves.len()
    );
    println!(
        "SUCCESS: Engine successfully played {} half-moves against itself",
        moves.len()
    );
}

#[test]
fn test_selfplay_threaded() {
    let mut white = EngineProcess::new();
    let mut black = EngineProcess::new();

    std::thread::sleep(Duration::from_millis(300));
    white.drain_output(Duration::from_millis(300));
    black.drain_output(Duration::from_millis(300));

    assert!(white.init_uci(), "White engine UCI handshake failed");
    assert!(black.init_uci(), "Black engine UCI handshake failed");

    let mut moves = Vec::new();
    let max_moves = 30;

    for i in 0..(max_moves * 2) {
        let current = if i % 2 == 0 { "White" } else { "Black" };
        let engine = if i % 2 == 0 {
            &mut white
        } else {
            &mut black
        };

        let position = if moves.is_empty() {
            "position startpos".to_string()
        } else {
            format!("position startpos moves {}", moves.join(" "))
        };

        let bestmove = engine.get_move_uci(&position, "go depth 4", Duration::from_secs(30));

        let bestmove = bestmove.unwrap_or_else(|| {
            panic!("{} failed to return move after {} half-moves", current, moves.len())
        });

        if bestmove == "0000" {
            println!("Game Over: {} has no moves", current);
            break;
        }

        moves.push(bestmove);

        if moves.len() >= 100 {
            println!("Game truncated at 50 full moves");
            break;
        }
    }

    assert!(
        moves.len() >= 10,
        "Game too short ({} moves)",
        moves.len()
    );
    println!("SUCCESS: Engine successfully played a coherent game");
}
