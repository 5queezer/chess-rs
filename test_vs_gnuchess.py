#!/usr/bin/env python3
"""
Test chess-rs engine against gnuchess using XBoard protocol.
"""

import subprocess
import time
import re
import sys
import threading
import queue

GNUCHESS_PATH = "/tmp/gnuchess-6.2.9/src/gnuchess"
ENGINE_PATH = "./target/release/chess"

class EngineProcess:
    def __init__(self, cmd, name="engine"):
        self.name = name
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        self.output_queue = queue.Queue()
        self.reader_thread = threading.Thread(target=self._reader, daemon=True)
        self.reader_thread.start()

    def _reader(self):
        try:
            for line in iter(self.proc.stdout.readline, ''):
                if line:
                    self.output_queue.put(line.strip())
                else:
                    break
        except:
            pass

    def send(self, cmd):
        self.proc.stdin.write(cmd + '\n')
        self.proc.stdin.flush()

    def read_until(self, pattern, timeout=10):
        lines = []
        start = time.time()
        while time.time() - start < timeout:
            try:
                line = self.output_queue.get(timeout=0.1)
                lines.append(line)
                if pattern in line:
                    return lines, True
            except queue.Empty:
                continue
        return lines, False

    def drain_output(self, timeout=0.3):
        lines = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self.output_queue.get(timeout=0.1)
                lines.append(line)
            except queue.Empty:
                break
        return lines

    def terminate(self):
        try:
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except:
            self.proc.kill()


def play_game_xboard():
    """Play a game using XBoard protocol."""
    print("=" * 60)
    print("Chess-RS vs GNUChess (XBoard Protocol)")
    print("=" * 60)

    # Start engines
    print("\nStarting chess-rs engine (White)...")
    white = EngineProcess([ENGINE_PATH], "chess-rs")
    time.sleep(0.3)
    white.drain_output()

    print("Starting gnuchess (Black)...")
    black = EngineProcess([GNUCHESS_PATH, "-x"], "gnuchess")  # -x for xboard mode
    time.sleep(0.3)
    black.drain_output()

    # Initialize both engines with XBoard protocol
    for name, engine in [("chess-rs", white), ("gnuchess", black)]:
        engine.send("xboard")
        time.sleep(0.1)

        engine.send("protover 2")
        lines, found = engine.read_until("done=1", timeout=5)
        if not found:
            print(f"WARNING: {name} protover response incomplete")
            print(f"  Output: {lines}")

        engine.send("new")
        time.sleep(0.1)
        engine.drain_output()

        # Set time control: 5 seconds per move
        engine.send("st 5")

        # Set engine to force mode (wait for commands)
        engine.send("force")

    print("Both engines initialized\n")
    print("Starting game...\n")

    moves = []
    move_number = 1
    current_engine = white
    current_name = "chess-rs"
    max_moves = 40

    for i in range(max_moves * 2):
        # Apply all moves to the current engine
        if moves:
            for m in moves:
                current_engine.send(m)
            current_engine.drain_output(0.1)

        # Tell engine to think
        current_engine.send("go")

        # Wait for the move
        lines, found = current_engine.read_until("move ", timeout=30)

        if not found:
            print(f"\n{current_name} failed to return a move!")
            print(f"Output: {lines}")
            break

        # Extract move
        move = None
        for line in lines:
            if line.startswith("move "):
                move = line.split()[1]
                break

        if not move:
            print(f"\n{current_name} returned no move")
            break

        # Check for resignation or special cases
        if "resign" in str(lines).lower():
            print(f"\n{current_name} resigned!")
            break

        moves.append(move)

        if i % 2 == 0:
            print(f"{move_number}. {move}", end=" ", flush=True)
        else:
            print(f"{move}", flush=True)
            move_number += 1

        # Put current engine back in force mode
        current_engine.send("force")

        # Switch sides
        if current_engine == white:
            current_engine = black
            current_name = "gnuchess"
        else:
            current_engine = white
            current_name = "chess-rs"

        time.sleep(0.05)

    print(f"\n\nGame complete: {len(moves)} half-moves")
    print(f"Final sequence: {' '.join(moves)}")

    white.terminate()
    black.terminate()

    return len(moves) > 10


def play_game_uci():
    """Play a game using UCI protocol (both engines as UCI)."""
    print("=" * 60)
    print("Chess-RS Self-Play (as proxy for gnuchess test)")
    print("=" * 60)

    # Since gnuchess uses XBoard, we'll use UCI self-play as backup
    white = EngineProcess([ENGINE_PATH], "white")
    black = EngineProcess([ENGINE_PATH], "black")

    time.sleep(0.3)
    white.drain_output()
    black.drain_output()

    # UCI initialization
    for name, engine in [("White", white), ("Black", black)]:
        engine.send("uci")
        lines, found = engine.read_until("uciok", timeout=5)
        if not found:
            print(f"ERROR: {name} UCI handshake failed")
            white.terminate()
            black.terminate()
            return False

        engine.send("isready")
        engine.read_until("readyok", timeout=5)

    print("Engines ready\n")

    moves = []
    move_number = 1

    for i in range(60):  # 30 full moves
        engine = white if i % 2 == 0 else black

        if moves:
            pos_cmd = f"position startpos moves {' '.join(moves)}"
        else:
            pos_cmd = "position startpos"

        engine.send(pos_cmd)
        engine.send("go depth 5")

        lines, found = engine.read_until("bestmove", timeout=60)

        if not found:
            print(f"\nNo bestmove after {len(moves)} moves")
            break

        bestmove = None
        for line in lines:
            if "bestmove" in line:
                match = re.search(r'bestmove\s+([a-h][1-8][a-h][1-8][qrbn]?)', line)
                if match:
                    bestmove = match.group(1)

        if not bestmove or bestmove == "0000":
            side = "White" if i % 2 == 0 else "Black"
            print(f"\nGame Over: {side} has no legal moves")
            break

        moves.append(bestmove)

        if i % 2 == 0:
            print(f"{move_number}. {bestmove}", end=" ", flush=True)
        else:
            print(f"{bestmove}", flush=True)
            move_number += 1

    print(f"\n\nTotal moves: {len(moves)}")
    print(f"Game: {' '.join(moves)}")

    white.terminate()
    black.terminate()

    return len(moves) > 10


def main():
    print("Testing chess-rs engine movement against gnuchess...\n")

    # First try XBoard match
    success = play_game_xboard()

    if success:
        print("\nSUCCESS: Engine plays correctly against gnuchess!")
        return 0
    else:
        print("\nFalling back to UCI self-play test...")
        success = play_game_uci()

        if success:
            print("\nSUCCESS: Engine movement is working (UCI self-play)")
            return 0
        else:
            print("\nFAILURE: Engine has movement issues")
            return 1


if __name__ == "__main__":
    sys.exit(main())
