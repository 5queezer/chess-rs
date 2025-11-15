#!/usr/bin/env python3
"""
Test chess-rs engine timing using UCI protocol.
Simpler and more reliable than XBoard protocol.
"""

import subprocess
import time
import sys
import re
from datetime import datetime

# Configuration
CHESS_RS = "./target/release/chess"
TIME_PER_MOVE_MS = 5000  # milliseconds
TIME_TOLERANCE_MS = 100  # Allow 100ms overhead
MAX_MOVES = 30
TIMEOUT_THRESHOLD_S = (TIME_PER_MOVE_MS / 1000) + 2

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

def start_engine():
    """Start the chess engine process."""
    log("Starting chess-rs...")
    proc = subprocess.Popen(
        [CHESS_RS],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    return proc

def send_command(proc, cmd):
    """Send command to engine."""
    log(f"  >> {cmd}")
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()

def read_line(proc, timeout=10):
    """Read a line from engine with timeout."""
    import select
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if ready:
        line = proc.stdout.readline().strip()
        if line:
            log(f"  << {line}")
        return line
    return None

def read_until_bestmove(proc, timeout=10):
    """Read lines until we get a bestmove or timeout."""
    start_time = time.time()
    best_move = None
    info_lines = []

    while time.time() - start_time < timeout:
        line = read_line(proc, timeout=1)
        if line is None:
            if proc.poll() is not None:
                log(f"  !! Engine process exited with code {proc.returncode}")
                break
            continue

        if line.startswith("bestmove"):
            parts = line.split()
            if len(parts) >= 2:
                best_move = parts[1]
            break
        elif line.startswith("info"):
            info_lines.append(line)

    elapsed = time.time() - start_time
    return best_move, elapsed, info_lines

def init_uci_engine(proc):
    """Initialize engine in UCI mode."""
    send_command(proc, "uci")

    # Wait for uciok
    start = time.time()
    while time.time() - start < 5:
        line = read_line(proc, timeout=0.5)
        if line == "uciok":
            break

    send_command(proc, "isready")

    # Wait for readyok
    start = time.time()
    while time.time() - start < 5:
        line = read_line(proc, timeout=0.5)
        if line == "readyok":
            break

    send_command(proc, "ucinewgame")
    time.sleep(0.1)

def play_game():
    """Test engine timing with UCI protocol."""
    log("=" * 60)
    log("Chess-rs Engine Timing Test (UCI Protocol)")
    log(f"Time per move: {TIME_PER_MOVE_MS}ms")
    log(f"Time tolerance: {TIME_TOLERANCE_MS}ms")
    log(f"Timeout threshold: {TIMEOUT_THRESHOLD_S}s")
    log("=" * 60)

    engine = start_engine()

    init_uci_engine(engine)
    log("Engine initialized successfully")

    moves = []  # List of moves played
    move_times = []  # List of (move_num, move, time_ms)
    slow_moves = []
    timeouts = []
    result = "unknown"

    # Alternate between playing as white and black using simple opening moves
    # We'll have the engine play both sides by alternating positions
    position = "startpos"
    move_list = []

    try:
        for move_num in range(1, MAX_MOVES + 1):
            log(f"\n--- Move {move_num} ---")

            # Set position
            if move_list:
                pos_cmd = f"position startpos moves {' '.join(move_list)}"
            else:
                pos_cmd = "position startpos"
            send_command(engine, pos_cmd)

            # Ask engine to search
            send_command(engine, f"go movetime {TIME_PER_MOVE_MS}")

            # Wait for bestmove
            best_move, elapsed, info_lines = read_until_bestmove(engine, timeout=TIMEOUT_THRESHOLD_S)

            if best_move is None:
                log(f"ERROR: Engine TIMEOUT after {TIMEOUT_THRESHOLD_S}s!")
                timeouts.append((move_num, elapsed))
                result = "timeout"
                break

            elapsed_ms = elapsed * 1000
            log(f"Engine played: {best_move} (took {elapsed_ms:.1f}ms)")
            move_times.append((move_num, best_move, elapsed_ms))

            if elapsed_ms > TIME_PER_MOVE_MS + TIME_TOLERANCE_MS:
                log(f"WARNING: Move took {elapsed_ms:.1f}ms, over {TIME_PER_MOVE_MS}ms + {TIME_TOLERANCE_MS}ms tolerance!")
                slow_moves.append((move_num, best_move, elapsed_ms))
            elif elapsed_ms > TIME_PER_MOVE_MS:
                log(f"NOTE: Move took {elapsed_ms:.1f}ms (within {TIME_TOLERANCE_MS}ms tolerance)")

            # Add move to list
            move_list.append(best_move)

            # Check for game end (0000 = no legal move)
            if best_move == "0000":
                log("Game ended (no legal move)")
                result = "game over"
                break

        if move_num >= MAX_MOVES:
            result = f"Reached {MAX_MOVES} moves"

    except Exception as e:
        log(f"ERROR: {e}")
        result = f"Error: {e}"
    finally:
        log("\nCleaning up...")
        send_command(engine, "quit")
        time.sleep(0.5)
        engine.terminate()

    # Print summary
    log("\n" + "=" * 60)
    log("TIMING TEST RESULTS")
    log("=" * 60)

    log(f"\nTotal moves: {len(move_times)}")
    log(f"Result: {result}")

    if move_times:
        times = [t[2] for t in move_times]
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)

        log(f"\nTiming statistics:")
        log(f"  Average time per move: {avg_time:.1f}ms")
        log(f"  Min time: {min_time:.1f}ms")
        log(f"  Max time: {max_time:.1f}ms")
        log(f"  Time limit: {TIME_PER_MOVE_MS}ms")

    if slow_moves:
        log(f"\nWARNING: {len(slow_moves)} moves exceeded time limit (+{TIME_TOLERANCE_MS}ms tolerance):")
        for num, move, t in slow_moves:
            log(f"  Move {num}: {move} took {t:.1f}ms")
    else:
        log(f"\nSUCCESS: All moves completed within {TIME_PER_MOVE_MS}ms + {TIME_TOLERANCE_MS}ms tolerance!")

    if timeouts:
        log(f"\nCRITICAL: {len(timeouts)} timeouts detected!")
        for num, t in timeouts:
            log(f"  Move {num}: timed out after {t:.3f}s")
        return False

    # Print move history
    log("\nMove history (first 10):")
    for num, move, t in move_times[:10]:
        log(f"  {num}. {move} ({t:.1f}ms)")
    if len(move_times) > 10:
        log(f"  ... ({len(move_times) - 10} more moves)")

    log("\n" + "=" * 60)

    passed = len(timeouts) == 0 and len(slow_moves) == 0
    if passed:
        log("TEST PASSED: Engine responds within time limits!")
    else:
        log("TEST FAILED: Timing issues detected!")

    return passed

if __name__ == "__main__":
    success = play_game()
    sys.exit(0 if success else 1)
