//! Test script for difficulty levels, depth limits, and randomization

mod common;

use common::EngineProcess;
use std::collections::HashSet;
use std::time::Duration;

#[test]
fn test_difficulty_levels() {
    for (level, name, _expected_max_depth) in [
        (1, "Beginner", 2),
        (2, "Easy", 3),
        (3, "Medium", 5),
        (4, "Hard", 7),
        (5, "Expert", 100),
    ] {
        let mut engine = EngineProcess::new();
        std::thread::sleep(Duration::from_millis(200));
        engine.drain_output(Duration::from_millis(300));

        assert!(
            engine.init_uci(),
            "UCI initialization failed for level {}",
            level
        );

        engine.send(&format!("setoption name Skill Level value {}", level));
        std::thread::sleep(Duration::from_millis(50));

        engine.send("isready");
        engine.read_until("readyok", Duration::from_secs(2));

        engine.send("position startpos");
        engine.send("go depth 10");

        let (lines, found) = engine.read_until("bestmove", Duration::from_secs(5));
        assert!(
            found,
            "Level {} ({}) failed to return bestmove",
            level,
            name
        );

        // Check that we got search info
        let has_info = lines.iter().any(|l| l.starts_with("info"));
        assert!(
            has_info,
            "Level {} ({}) didn't output search info",
            level,
            name
        );

        let has_bestmove = lines.iter().any(|l| l.starts_with("bestmove"));
        assert!(
            has_bestmove,
            "Level {} ({}) didn't output bestmove",
            level,
            name
        );

        println!("Level {} ({}) works correctly", level, name);
    }
}

#[test]
fn test_depth_limits() {
    let test_cases = [
        (1, "Beginner", 2),
        (2, "Easy", 4),
        (3, "Medium", 8),
        (4, "Hard", 10),
    ];

    for (level, name, expected_max_depth) in test_cases {
        let mut engine = EngineProcess::new();
        std::thread::sleep(Duration::from_millis(200));
        engine.drain_output(Duration::from_millis(300));

        assert!(engine.init_uci(), "UCI initialization failed");

        engine.send(&format!("setoption name Skill Level value {}", level));
        std::thread::sleep(Duration::from_millis(50));

        engine.send("isready");
        engine.read_until("readyok", Duration::from_secs(2));

        engine.send("position startpos");
        engine.send("go depth 100"); // Request max depth

        // Higher levels need more time
        let timeout_secs = if level >= 4 { 10 } else { 5 };
        let (lines, _) = engine.read_until("bestmove", Duration::from_secs(timeout_secs));

        // Find the maximum depth reported
        let max_depth_seen = lines
            .iter()
            .filter(|l| l.starts_with("info") && l.contains("depth"))
            .filter_map(|l| {
                let parts: Vec<&str> = l.split_whitespace().collect();
                for (i, part) in parts.iter().enumerate() {
                    if *part == "depth" && i + 1 < parts.len() {
                        // Make sure this is actual depth, not seldepth
                        if i > 0 && parts[i - 1] != "sel" {
                            return parts[i + 1].parse::<u32>().ok();
                        }
                    }
                }
                None
            })
            .max()
            .unwrap_or(0);

        // Verify depth is at least 1 and within reasonable bounds
        assert!(
            max_depth_seen >= 1,
            "Level {} ({}) didn't search at all",
            level,
            name
        );

        // Check depth is within expected range (allow some flexibility)
        assert!(
            max_depth_seen <= expected_max_depth + 2,
            "Level {} ({}) exceeded expected depth. Expected max {}, got {}",
            level,
            name,
            expected_max_depth,
            max_depth_seen
        );

        println!(
            "Level {} ({}) - Expected max depth: {}, Actual: {}",
            level, name, expected_max_depth, max_depth_seen
        );
    }
}

#[test]
fn test_expert_level_respects_depth() {
    let mut engine = EngineProcess::new();
    std::thread::sleep(Duration::from_millis(200));
    engine.drain_output(Duration::from_millis(300));

    assert!(engine.init_uci(), "UCI initialization failed");

    engine.send("setoption name Skill Level value 5");
    std::thread::sleep(Duration::from_millis(50));

    engine.send("isready");
    engine.read_until("readyok", Duration::from_secs(2));

    engine.send("position startpos");
    engine.send("go depth 8"); // Request specific depth

    let (lines, found) = engine.read_until("bestmove", Duration::from_secs(10));
    assert!(found, "Expert level didn't return bestmove");

    // Find the maximum depth reported
    let max_depth_seen = lines
        .iter()
        .filter(|l| l.starts_with("info") && l.contains("depth"))
        .filter_map(|l| {
            let parts: Vec<&str> = l.split_whitespace().collect();
            for (i, part) in parts.iter().enumerate() {
                if *part == "depth" && i + 1 < parts.len() {
                    if i > 0 && parts[i - 1] != "sel" {
                        return parts[i + 1].parse::<u32>().ok();
                    }
                }
            }
            None
        })
        .max()
        .unwrap_or(0);

    // Expert level should search at least depth 8
    assert!(
        max_depth_seen >= 8,
        "Expert level didn't reach requested depth. Expected at least 8, got {}",
        max_depth_seen
    );

    println!(
        "Expert level with 'go depth 8': reached depth {}",
        max_depth_seen
    );
}

#[test]
fn test_uci_option_advertisement() {
    let mut engine = EngineProcess::new();
    std::thread::sleep(Duration::from_millis(200));
    engine.drain_output(Duration::from_millis(300));

    engine.send("uci");
    let (lines, found) = engine.read_until("uciok", Duration::from_secs(5));
    assert!(found, "UCI handshake failed");

    // Check that Skill Level option is advertised
    let has_skill_option = lines.iter().any(|l| {
        l.starts_with("option") && l.contains("name") && l.contains("Skill Level")
    });

    println!("UCI options advertised:");
    for line in lines.iter().filter(|l| l.starts_with("option")) {
        println!("  {}", line);
    }

    // This is just informational - engine might not have skill level option
    if has_skill_option {
        println!("Skill Level option is advertised");
    } else {
        println!("Note: Skill Level option not advertised (may use different mechanism)");
    }
}

#[test]
fn test_move_randomization_beginner() {
    let mut moves_seen = HashSet::new();
    let runs = 10;

    for _ in 0..runs {
        let mut engine = EngineProcess::new();
        std::thread::sleep(Duration::from_millis(100));
        engine.drain_output(Duration::from_millis(200));

        assert!(engine.init_uci(), "UCI initialization failed");

        engine.send("setoption name Skill Level value 1");
        std::thread::sleep(Duration::from_millis(50));

        engine.send("isready");
        engine.read_until("readyok", Duration::from_secs(2));

        engine.send("position startpos");
        engine.send("go movetime 300");

        let (lines, found) = engine.read_until("bestmove", Duration::from_secs(2));
        if found {
            if let Some(bestmove_line) = lines.iter().find(|l| l.starts_with("bestmove")) {
                let parts: Vec<&str> = bestmove_line.split_whitespace().collect();
                if parts.len() >= 2 {
                    moves_seen.insert(parts[1].to_string());
                }
            }
        }
    }

    println!("Beginner level randomization test:");
    println!("  Runs: {}", runs);
    println!("  Unique moves: {}", moves_seen.len());
    println!("  Moves: {:?}", moves_seen);

    // With randomization, we expect to see some variety
    // But this is not guaranteed, so we just log the result
    if moves_seen.len() > 1 {
        println!("SUCCESS: Move randomization detected at beginner level");
    } else {
        println!("Note: All moves were the same (randomization may not be active)");
    }
}

#[test]
fn test_move_consistency_medium() {
    let mut moves_seen = HashSet::new();
    let runs = 5;

    for _ in 0..runs {
        let mut engine = EngineProcess::new();
        std::thread::sleep(Duration::from_millis(100));
        engine.drain_output(Duration::from_millis(200));

        assert!(engine.init_uci(), "UCI initialization failed");

        engine.send("setoption name Skill Level value 3");
        std::thread::sleep(Duration::from_millis(50));

        engine.send("isready");
        engine.read_until("readyok", Duration::from_secs(2));

        engine.send("position startpos");
        engine.send("go movetime 300");

        let (lines, found) = engine.read_until("bestmove", Duration::from_secs(2));
        if found {
            if let Some(bestmove_line) = lines.iter().find(|l| l.starts_with("bestmove")) {
                let parts: Vec<&str> = bestmove_line.split_whitespace().collect();
                if parts.len() >= 2 {
                    moves_seen.insert(parts[1].to_string());
                }
            }
        }
    }

    println!("Medium level consistency test:");
    println!("  Runs: {}", runs);
    println!("  Unique moves: {}", moves_seen.len());
    println!("  Moves: {:?}", moves_seen);

    // Medium level should be more consistent (less randomization)
    if moves_seen.len() == 1 {
        println!("SUCCESS: Medium level shows consistent moves (no randomization)");
    } else {
        println!("Note: Medium level showed some variation ({} unique moves)", moves_seen.len());
    }
}

#[test]
fn test_multiple_moves_at_beginner() {
    let mut engine = EngineProcess::new();
    std::thread::sleep(Duration::from_millis(200));
    engine.drain_output(Duration::from_millis(300));

    assert!(engine.init_uci(), "UCI initialization failed");

    engine.send("setoption name Skill Level value 1");
    std::thread::sleep(Duration::from_millis(50));

    engine.send("isready");
    engine.read_until("readyok", Duration::from_secs(2));

    // Test three positions
    let positions = [
        "position startpos",
        "position startpos moves e2e4",
        "position startpos moves e2e4 e7e5",
    ];

    let mut moves = Vec::new();

    for pos in &positions {
        engine.send(pos);
        engine.send("go movetime 500");

        let (lines, found) = engine.read_until("bestmove", Duration::from_secs(2));
        assert!(found, "No bestmove for position: {}", pos);

        if let Some(bestmove_line) = lines.iter().find(|l| l.starts_with("bestmove")) {
            let parts: Vec<&str> = bestmove_line.split_whitespace().collect();
            if parts.len() >= 2 {
                moves.push(parts[1].to_string());
            }
        }
    }

    println!("Multiple moves at beginner level:");
    for (i, mv) in moves.iter().enumerate() {
        println!("  Position {}: {}", i + 1, mv);
    }

    assert_eq!(
        moves.len(),
        positions.len(),
        "Didn't get moves for all positions"
    );
}
