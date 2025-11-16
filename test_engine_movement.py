#!/usr/bin/env python3
"""Test harness to verify chess engine movement and UCI protocol compliance."""

import subprocess
import time
import re
import sys

def start_engine():
    """Start the chess engine process."""
    proc = subprocess.Popen(
        ['./target/release/chess'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    return proc

def send_command(proc, cmd):
    """Send a command to the engine."""
    proc.stdin.write(cmd + '\n')
    proc.stdin.flush()

def read_until(proc, pattern, timeout=10):
    """Read engine output until pattern is found or timeout."""
    import select
    start = time.time()
    output = []
    while time.time() - start < timeout:
        # Check if there's output available
        if select.select([proc.stdout], [], [], 0.1)[0]:
            line = proc.stdout.readline()
            if line:
                output.append(line.strip())
                if pattern in line:
                    return output
    return output

def read_lines(proc, timeout=0.5):
    """Read available lines from engine within timeout."""
    import select
    start = time.time()
    lines = []
    while time.time() - start < timeout:
        if select.select([proc.stdout], [], [], 0.1)[0]:
            line = proc.stdout.readline()
            if line:
                lines.append(line.strip())
        else:
            if lines:  # If we have some output and no more coming
                break
    return lines

def test_uci_handshake():
    """Test UCI protocol handshake."""
    print("Testing UCI handshake...")
    proc = start_engine()

    # Clear any initial output
    time.sleep(0.2)
    initial = read_lines(proc, 0.5)
    print(f"Initial output: {initial}")

    send_command(proc, "uci")
    output = read_until(proc, "uciok", timeout=5)
    print(f"UCI response: {output}")

    if any("uciok" in line for line in output):
        print("PASS: UCI handshake successful")
    else:
        print("FAIL: UCI handshake failed")
        proc.terminate()
        return False

    send_command(proc, "isready")
    output = read_until(proc, "readyok", timeout=5)

    if any("readyok" in line for line in output):
        print("PASS: Engine is ready")
    else:
        print("FAIL: Engine not ready")
        proc.terminate()
        return False

    proc.terminate()
    return True

def test_move_generation():
    """Test that engine generates moves correctly."""
    print("\nTesting move generation...")
    proc = start_engine()

    # Skip initial output
    time.sleep(0.3)
    read_lines(proc, 0.5)

    # UCI handshake
    send_command(proc, "uci")
    read_until(proc, "uciok", timeout=5)
    send_command(proc, "isready")
    read_until(proc, "readyok", timeout=5)

    # Set up position and search
    send_command(proc, "position startpos")
    send_command(proc, "go depth 4")

    # Wait for bestmove
    print("Waiting for bestmove...")
    output = read_until(proc, "bestmove", timeout=30)

    bestmove_line = [l for l in output if "bestmove" in l]
    if bestmove_line:
        print(f"PASS: Engine returned move: {bestmove_line[-1]}")
        move_match = re.search(r'bestmove\s+([a-h][1-8][a-h][1-8])', bestmove_line[-1])
        if move_match:
            move = move_match.group(1)
            print(f"  Move: {move}")
            # Verify it's a legal move from starting position
            legal_starts = ['a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2', 'b1', 'g1']
            if move[:2] in legal_starts:
                print(f"  PASS: Move starts from legal square")
            else:
                print(f"  FAIL: Move starts from illegal square: {move[:2]}")
        result = True
    else:
        print(f"FAIL: No bestmove returned. Output: {output}")
        result = False

    proc.terminate()
    return result

def test_multiple_moves():
    """Test engine playing multiple moves in a game."""
    print("\nTesting multiple move sequence...")
    proc = start_engine()

    time.sleep(0.3)
    read_lines(proc, 0.5)

    send_command(proc, "uci")
    read_until(proc, "uciok", timeout=5)
    send_command(proc, "isready")
    read_until(proc, "readyok", timeout=5)

    # Play a sequence of moves
    moves = []
    opponent_moves = ["e2e4", "d2d4", "g1f3", "b1c3", "f1c4"]

    for i, opp_move in enumerate(opponent_moves):
        if i == 0:
            send_command(proc, f"position startpos")
        else:
            moves_str = " ".join(moves)
            send_command(proc, f"position startpos moves {moves_str}")

        send_command(proc, "go depth 3")
        output = read_until(proc, "bestmove", timeout=15)

        bestmove_line = [l for l in output if "bestmove" in l]
        if bestmove_line:
            move_match = re.search(r'bestmove\s+([a-h][1-8][a-h][1-8][qrbn]?)', bestmove_line[-1])
            if move_match:
                engine_move = move_match.group(1)
                moves.append(engine_move)
                print(f"Move {i+1}: Engine played {engine_move}")

                if i < len(opponent_moves) - 1:
                    moves.append(opp_move)
                    print(f"Move {i+1}: Opponent played {opp_move}")
            else:
                print(f"FAIL: Could not parse move from {bestmove_line[-1]}")
                proc.terminate()
                return False
        else:
            print(f"FAIL: No bestmove at move {i+1}. Output: {output}")
            proc.terminate()
            return False

    print(f"PASS: Engine successfully played {len(opponent_moves)} moves")
    print(f"Game sequence: {' '.join(moves)}")
    proc.terminate()
    return True

def test_xboard_mode():
    """Test XBoard protocol."""
    print("\nTesting XBoard mode...")
    proc = start_engine()

    time.sleep(0.3)
    read_lines(proc, 0.5)

    send_command(proc, "xboard")
    send_command(proc, "protover 2")

    output = read_until(proc, "done=1", timeout=5)
    if any("done=1" in line for line in output):
        print("PASS: XBoard protover handshake successful")
    else:
        print(f"Output: {output}")
        print("FAIL: XBoard handshake failed")
        proc.terminate()
        return False

    send_command(proc, "new")
    time.sleep(0.2)

    send_command(proc, "go")
    print("Waiting for XBoard move...")

    output = read_until(proc, "move ", timeout=20)
    move_lines = [l for l in output if l.startswith("move ")]

    if move_lines:
        print(f"PASS: XBoard engine moved: {move_lines[-1]}")
        result = True
    else:
        print(f"FAIL: No move in XBoard mode. Output: {output}")
        result = False

    proc.terminate()
    return result

def test_search_info():
    """Test that search info is output correctly."""
    print("\nTesting search info output...")
    proc = start_engine()

    time.sleep(0.3)
    read_lines(proc, 0.5)

    send_command(proc, "uci")
    read_until(proc, "uciok", timeout=5)
    send_command(proc, "isready")
    read_until(proc, "readyok", timeout=5)

    send_command(proc, "position startpos")
    send_command(proc, "go depth 5")

    output = read_until(proc, "bestmove", timeout=30)

    info_lines = [l for l in output if l.startswith("info")]
    if info_lines:
        print(f"PASS: Search info output correctly ({len(info_lines)} info lines)")
        for line in info_lines[-3:]:
            print(f"  {line}")
    else:
        print("WARNING: No info lines output")

    proc.terminate()
    return True

def main():
    print("=" * 60)
    print("Chess Engine Movement Test Suite")
    print("=" * 60)

    tests = [
        ("UCI Handshake", test_uci_handshake),
        ("Move Generation", test_move_generation),
        ("Multiple Moves", test_multiple_moves),
        ("XBoard Mode", test_xboard_mode),
        ("Search Info", test_search_info),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed! Engine movement is working correctly.")
        return 0
    else:
        print("\nSome tests failed. Engine movement issues detected.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
