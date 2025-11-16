#!/usr/bin/env python3
"""
Self-play test: Have the chess engine play against itself via UCI protocol.
This tests the engine's ability to maintain a coherent game.
"""

import subprocess
import time
import re
import sys
import select

def start_engine():
    """Start the chess engine process."""
    proc = subprocess.Popen(
        ['stdbuf', '-oL', './target/release/chess'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0  # Unbuffered
    )
    return proc

def send(proc, cmd):
    """Send a command to the engine."""
    proc.stdin.write(cmd + '\n')
    proc.stdin.flush()

def read_until_bestmove(proc, timeout=30):
    """Read engine output until bestmove or timeout."""
    start = time.time()
    output = []
    bestmove = None

    while time.time() - start < timeout:
        if select.select([proc.stdout], [], [], 0.1)[0]:
            line = proc.stdout.readline().strip()
            if line:
                output.append(line)
                if line.startswith("bestmove"):
                    match = re.search(r'bestmove\s+([a-h][1-8][a-h][1-8][qrbn]?)', line)
                    if match:
                        bestmove = match.group(1)
                    break
    return bestmove, output

def read_available(proc, timeout=0.5):
    """Read any available output."""
    start = time.time()
    lines = []
    while time.time() - start < timeout:
        if select.select([proc.stdout], [], [], 0.1)[0]:
            line = proc.stdout.readline().strip()
            if line:
                lines.append(line)
                # Keep reading if there might be more
                start = time.time()
        elif lines:
            break
    return lines

def read_until_pattern(proc, pattern, timeout=5):
    """Read until we see a specific pattern."""
    start = time.time()
    lines = []
    while time.time() - start < timeout:
        if select.select([proc.stdout], [], [], 0.1)[0]:
            line = proc.stdout.readline().strip()
            if line:
                lines.append(line)
                if pattern in line:
                    return lines, True
    return lines, False

def is_game_over(moves):
    """Simple check for game-over conditions."""
    # 50-move rule (simplified)
    if len(moves) >= 100:
        return True, "50-move rule"
    # Repetition detection (simplified - just check for long games)
    if len(moves) >= 200:
        return True, "Too many moves"
    return False, ""

def main():
    print("=" * 60)
    print("Chess Engine Self-Play Test")
    print("=" * 60)

    # Start two engine instances
    print("Starting White engine...")
    white = start_engine()
    time.sleep(0.3)
    read_available(white)

    print("Starting Black engine...")
    black = start_engine()
    time.sleep(0.3)
    read_available(black)

    # UCI handshake for both
    for name, engine in [("White", white), ("Black", black)]:
        send(engine, "uci")
        lines, found = read_until_pattern(engine, "uciok", timeout=5)
        if not found:
            print(f"ERROR: {name} engine UCI handshake failed")
            print(f"Output received: {lines}")
            white.terminate()
            black.terminate()
            return 1

        send(engine, "isready")
        lines, found = read_until_pattern(engine, "readyok", timeout=5)
        if not found:
            print(f"ERROR: {name} engine not ready")
            print(f"Output received: {lines}")
            white.terminate()
            black.terminate()
            return 1

    print("Both engines initialized successfully")
    print("\nStarting self-play game...\n")

    moves = []
    move_number = 1
    max_moves = 50  # Play up to 50 moves each side

    for i in range(max_moves * 2):
        current_side = "White" if i % 2 == 0 else "Black"
        engine = white if i % 2 == 0 else black

        # Set up position
        if moves:
            position_cmd = f"position startpos moves {' '.join(moves)}"
        else:
            position_cmd = "position startpos"

        send(engine, position_cmd)
        send(engine, "go depth 4")

        # Get the move
        bestmove, output = read_until_bestmove(engine, timeout=30)

        if not bestmove:
            print(f"\nERROR: {current_side} failed to return a move!")
            print(f"Output: {output}")
            print(f"\nGame ended after {len(moves)} half-moves")
            break

        # Check for special cases
        if bestmove == "0000" or bestmove == "(none)":
            print(f"\nGame Over: {current_side} has no legal moves (checkmate or stalemate)")
            break

        moves.append(bestmove)

        if i % 2 == 0:
            print(f"{move_number}. {bestmove}", end=" ")
        else:
            print(f"{bestmove}")
            move_number += 1

        # Check for game over conditions
        over, reason = is_game_over(moves)
        if over:
            print(f"\nGame Over: {reason}")
            break

        # Small delay to avoid overwhelming
        time.sleep(0.05)

    print(f"\n\nGame finished after {len(moves)} half-moves")
    print(f"Full move sequence: {' '.join(moves)}")

    # Cleanup
    white.terminate()
    black.terminate()

    # Validate the game was coherent
    if len(moves) > 10:
        print("\nSUCCESS: Engine successfully played multiple moves against itself")
        return 0
    else:
        print("\nWARNING: Game was very short, might indicate issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
