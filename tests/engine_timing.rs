//! Test chess-rs engine timing using UCI protocol

mod common;

use common::EngineProcess;
use std::time::{Duration, Instant};

const TIME_PER_MOVE_MS: u64 = 5000;
const TIME_TOLERANCE_MS: u64 = 100;
const MAX_MOVES: usize = 30;
const TIMEOUT_THRESHOLD_SECS: u64 = (TIME_PER_MOVE_MS / 1000) + 2;

struct MoveTime {
    move_num: usize,
    mv: String,
    elapsed_ms: u64,
}

#[test]
fn test_engine_timing_uci() {
    let mut engine = EngineProcess::new();
    std::thread::sleep(Duration::from_millis(300));
    engine.drain_output(Duration::from_millis(500));

    assert!(engine.init_uci(), "UCI initialization failed");
    engine.send("ucinewgame");
    std::thread::sleep(Duration::from_millis(100));

    let mut move_list: Vec<String> = Vec::new();
    let mut move_times: Vec<MoveTime> = Vec::new();
    let mut slow_moves: Vec<MoveTime> = Vec::new();
    let mut timeouts: Vec<(usize, f64)> = Vec::new();

    for move_num in 1..=MAX_MOVES {
        let pos_cmd = if move_list.is_empty() {
            "position startpos".to_string()
        } else {
            format!("position startpos moves {}", move_list.join(" "))
        };

        engine.send(&pos_cmd);
        engine.send(&format!("go movetime {}", TIME_PER_MOVE_MS));

        let start = Instant::now();
        let (lines, found) = engine.read_until(
            "bestmove",
            Duration::from_secs(TIMEOUT_THRESHOLD_SECS),
        );
        let elapsed = start.elapsed();

        if !found {
            timeouts.push((move_num, elapsed.as_secs_f64()));
            break;
        }

        let best_move = lines
            .iter()
            .rev()
            .find_map(|line| {
                if line.starts_with("bestmove") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        return Some(parts[1].to_string());
                    }
                }
                None
            })
            .expect("No bestmove found in output");

        let elapsed_ms = elapsed.as_millis() as u64;

        let move_time = MoveTime {
            move_num,
            mv: best_move.clone(),
            elapsed_ms,
        };

        if elapsed_ms > TIME_PER_MOVE_MS + TIME_TOLERANCE_MS {
            slow_moves.push(MoveTime {
                move_num,
                mv: best_move.clone(),
                elapsed_ms,
            });
        }

        move_times.push(move_time);

        // Check for game end
        if best_move == "0000" {
            break;
        }

        move_list.push(best_move);
    }

    // Analyze results
    assert!(
        timeouts.is_empty(),
        "Engine timed out on moves: {:?}",
        timeouts
    );

    assert!(
        slow_moves.is_empty(),
        "Moves exceeded time limit ({}ms + {}ms tolerance): {:?}",
        TIME_PER_MOVE_MS,
        TIME_TOLERANCE_MS,
        slow_moves
            .iter()
            .map(|m| format!("Move {}: {} took {}ms", m.move_num, m.mv, m.elapsed_ms))
            .collect::<Vec<_>>()
    );

    // Print timing statistics
    if !move_times.is_empty() {
        let times: Vec<u64> = move_times.iter().map(|m| m.elapsed_ms).collect();
        let avg_time = times.iter().sum::<u64>() as f64 / times.len() as f64;
        let max_time = *times.iter().max().unwrap();
        let min_time = *times.iter().min().unwrap();

        println!("Timing statistics:");
        println!("  Total moves: {}", move_times.len());
        println!("  Average time per move: {:.1}ms", avg_time);
        println!("  Min time: {}ms", min_time);
        println!("  Max time: {}ms", max_time);
        println!("  Time limit: {}ms", TIME_PER_MOVE_MS);
    }

    println!("SUCCESS: All moves completed within time limits!");
}

#[test]
fn test_timing_consistency() {
    let mut engine = EngineProcess::new();
    std::thread::sleep(Duration::from_millis(300));
    engine.drain_output(Duration::from_millis(500));

    assert!(engine.init_uci(), "UCI initialization failed");

    // Test 5 moves with the same time control
    let test_time_ms = 1000;
    let mut times = Vec::new();

    for _ in 0..5 {
        engine.send("position startpos");
        engine.send(&format!("go movetime {}", test_time_ms));

        let start = Instant::now();
        let (_, found) = engine.read_until("bestmove", Duration::from_secs(5));
        let elapsed = start.elapsed();

        assert!(found, "No bestmove returned");
        times.push(elapsed.as_millis() as u64);
    }

    // Check that all times are reasonably consistent
    let avg = times.iter().sum::<u64>() as f64 / times.len() as f64;
    let variance: f64 = times
        .iter()
        .map(|t| {
            let diff = *t as f64 - avg;
            diff * diff
        })
        .sum::<f64>()
        / times.len() as f64;
    let std_dev = variance.sqrt();

    println!("Time consistency test:");
    println!("  Times: {:?}", times);
    println!("  Average: {:.1}ms", avg);
    println!("  Std Dev: {:.1}ms", std_dev);

    // Standard deviation should be reasonable (< 500ms for 1000ms searches)
    assert!(
        std_dev < 500.0,
        "Timing too inconsistent: std_dev = {:.1}ms",
        std_dev
    );
}
