#!/usr/bin/env python3
"""
Test chess-rs engine timing by running it against GNU Chess.
Uses XBoard/CECP protocol to manage the game.
"""

import subprocess
import time
import sys
import re
from datetime import datetime

# Configuration
CHESS_RS = "./target/release/chess"
GNUCHESS = "/tmp/gnuchess_extracted/usr/games/gnuchess"
TIME_PER_MOVE = 5  # seconds
TIME_TOLERANCE = 0.05  # Allow 50ms overhead for thread operations
MAX_MOVES = 50     # max moves per side
TIMEOUT_THRESHOLD = TIME_PER_MOVE + 2  # Allow some buffer

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

def start_engine(path, name):
    """Start an engine process."""
    log(f"Starting {name}...")
    proc = subprocess.Popen(
        [path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0  # Unbuffered
    )
    return proc

def send_command(proc, cmd, name="engine"):
    """Send command to engine."""
    log(f"  >> {name}: {cmd}")
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()

def read_line(proc, name="engine", timeout=10):
    """Read a line from engine with timeout."""
    import select
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if ready:
        line = proc.stdout.readline().strip()
        if line:
            log(f"  << {name}: {line}")
        return line
    return None

def read_until_move(proc, name="engine", timeout=10):
    """Read lines until we get a move or timeout."""
    start_time = time.time()
    move = None

    while time.time() - start_time < timeout:
        line = read_line(proc, name, timeout=1)
        if line is None:
            # Check if process is still running
            if proc.poll() is not None:
                log(f"  !! {name} process exited with code {proc.returncode}")
                # Try to read any stderr
                try:
                    stderr = proc.stderr.read()
                    if stderr:
                        log(f"  !! {name} stderr: {stderr[:500]}")
                except:
                    pass
                break
            continue

        # XBoard move format: move <move> or just <move>
        if line.startswith("move "):
            move = line.split()[1]
            break
        # Check for single move (letter + number format)
        elif re.match(r'^[a-h][1-8][a-h][1-8][qrbn]?$', line):
            move = line
            break
        # GNU Chess might output just the move
        elif re.match(r'^[a-h][1-8][a-h][1-8][qrbn]?$', line.split()[-1] if line.split() else ""):
            move = line.split()[-1]
            break

    elapsed = time.time() - start_time
    return move, elapsed

def init_xboard_engine(proc, name):
    """Initialize engine in XBoard mode."""
    send_command(proc, "xboard", name)
    time.sleep(0.1)
    send_command(proc, "protover 2", name)

    # Read feature announcements and parse them
    features = {}
    start = time.time()
    while time.time() - start < 2:
        line = read_line(proc, name, timeout=0.5)
        if line and "done=1" in line:
            break
        if line and line.startswith("feature "):
            # Parse feature=value pairs
            for part in line.split():
                if "=" in part:
                    key, val = part.split("=", 1)
                    features[key] = val

    send_command(proc, "new", name)
    send_command(proc, f"st {TIME_PER_MOVE}", name)  # Time per move in seconds
    time.sleep(0.2)

    return features

def play_game():
    """Play a game between chess-rs (white) and GNU Chess (black)."""
    log("=" * 60)
    log("Starting engine timing test")
    log(f"Time per move: {TIME_PER_MOVE} seconds")
    log(f"Timeout threshold: {TIMEOUT_THRESHOLD} seconds")
    log("=" * 60)

    # Start both engines
    chess_rs = start_engine(CHESS_RS, "chess-rs")
    gnuchess = start_engine(GNUCHESS, "gnuchess")

    # Initialize both in XBoard mode
    chess_rs_features = init_xboard_engine(chess_rs, "chess-rs")
    gnuchess_features = init_xboard_engine(gnuchess, "gnuchess")

    log(f"chess-rs features: usermove={chess_rs_features.get('usermove', '0')}")
    log(f"gnuchess features: usermove={gnuchess_features.get('usermove', '0')}")

    # Determine if usermove prefix is needed
    gnuchess_usermove = gnuchess_features.get("usermove", "0") == "1"

    # Put GNU Chess in force mode (accepts moves, doesn't play)
    send_command(gnuchess, "force", "gnuchess")

    moves = []
    timeouts = []
    slow_moves = []
    game_over = False
    result = "unknown"

    try:
        # First move: explicitly tell chess-rs to go
        log("\n--- Move 1 ---")
        log("chess-rs thinking...")
        send_command(chess_rs, "go", "chess-rs")

        for move_num in range(1, MAX_MOVES + 1):
            # Wait for chess-rs move
            move, elapsed = read_until_move(chess_rs, "chess-rs", timeout=TIMEOUT_THRESHOLD)

            if move is None:
                log(f"ERROR: chess-rs TIMEOUT after {TIMEOUT_THRESHOLD}s!")
                timeouts.append(("chess-rs", move_num, elapsed))
                result = "chess-rs timeout"
                break

            log(f"chess-rs played: {move} (took {elapsed:.3f}s)")
            moves.append(("chess-rs", move, elapsed))

            if elapsed > TIME_PER_MOVE + TIME_TOLERANCE:
                log(f"WARNING: Move took {elapsed:.3f}s, over {TIME_PER_MOVE}s limit!")
                slow_moves.append(("chess-rs", move_num, move, elapsed))
            elif elapsed > TIME_PER_MOVE:
                log(f"NOTE: Move took {elapsed:.3f}s (within {TIME_TOLERANCE}s tolerance)")

            # Send chess-rs's move to GNU Chess (use usermove prefix if required)
            if gnuchess_usermove:
                send_command(gnuchess, f"usermove {move}", "gnuchess")
            else:
                send_command(gnuchess, move, "gnuchess")
            time.sleep(0.1)

            # GNU Chess responds
            log("gnuchess thinking...")
            send_command(gnuchess, "go", "gnuchess")
            gnuchess_move, elapsed = read_until_move(gnuchess, "gnuchess", timeout=TIMEOUT_THRESHOLD)

            if gnuchess_move is None:
                log(f"gnuchess did not respond (might be game over)")
                result = "gnuchess no response (possible checkmate/stalemate)"
                break

            log(f"gnuchess played: {gnuchess_move} (took {elapsed:.3f}s)")
            moves.append(("gnuchess", gnuchess_move, elapsed))

            # Put GNU Chess back in force mode
            send_command(gnuchess, "force", "gnuchess")

            # Send gnuchess's move to chess-rs
            # chess-rs should auto-respond since it's not in force mode
            log(f"\n--- Move {move_num + 1} ---")
            log("chess-rs thinking...")
            send_command(chess_rs, gnuchess_move, "chess-rs")
            # chess-rs will auto-respond because it's not in force mode

        if move_num >= MAX_MOVES:
            result = f"Game reached {MAX_MOVES} moves limit"

    except Exception as e:
        log(f"ERROR: {e}")
        result = f"Error: {e}"
    finally:
        # Cleanup
        log("\nCleaning up...")
        send_command(chess_rs, "quit", "chess-rs")
        send_command(gnuchess, "quit", "gnuchess")
        time.sleep(0.5)
        chess_rs.terminate()
        gnuchess.terminate()

    # Print summary
    log("\n" + "=" * 60)
    log("TIMING TEST RESULTS")
    log("=" * 60)

    log(f"\nTotal moves played: {len(moves)}")
    log(f"Result: {result}")

    # Analyze chess-rs timing
    chess_rs_times = [m[2] for m in moves if m[0] == "chess-rs"]
    if chess_rs_times:
        avg_time = sum(chess_rs_times) / len(chess_rs_times)
        max_time = max(chess_rs_times)
        min_time = min(chess_rs_times)

        log(f"\nchess-rs timing statistics:")
        log(f"  Average time per move: {avg_time:.3f}s")
        log(f"  Min time: {min_time:.3f}s")
        log(f"  Max time: {max_time:.3f}s")
        log(f"  Time limit: {TIME_PER_MOVE}s")

    if slow_moves:
        log(f"\nWARNING: {len(slow_moves)} moves exceeded time limit (>{TIME_PER_MOVE}s + {TIME_TOLERANCE}s tolerance):")
        for engine, num, move, t in slow_moves:
            log(f"  Move {num}: {engine} played {move} in {t:.3f}s")
    else:
        log(f"\nSUCCESS: All chess-rs moves completed within {TIME_PER_MOVE}s limit (with {TIME_TOLERANCE}s tolerance)!")

    if timeouts:
        log(f"\nCRITICAL: {len(timeouts)} timeouts detected!")
        for engine, num, t in timeouts:
            log(f"  Move {num}: {engine} timed out")
        return False

    # Print move history (first 10 and last 5)
    log("\nMove history (showing first 10 and last 5):")
    if len(moves) <= 15:
        for i, (engine, move, t) in enumerate(moves):
            side = "W" if engine == "chess-rs" else "B"
            log(f"  {i+1}. [{side}] {move} ({t:.3f}s)")
    else:
        for i, (engine, move, t) in enumerate(moves[:10]):
            side = "W" if engine == "chess-rs" else "B"
            log(f"  {i+1}. [{side}] {move} ({t:.3f}s)")
        log(f"  ... ({len(moves) - 15} moves omitted) ...")
        for i, (engine, move, t) in enumerate(moves[-5:], start=len(moves)-4):
            side = "W" if engine == "chess-rs" else "B"
            log(f"  {i}. [{side}] {move} ({t:.3f}s)")

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
