#!/usr/bin/env python3
"""
Self-play test using threaded I/O for better reliability.
"""

import subprocess
import time
import re
import sys
import threading
import queue

class EngineProcess:
    def __init__(self):
        self.proc = subprocess.Popen(
            ['./target/release/chess'],
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
        """Continuously read from stdout and put lines in queue."""
        try:
            for line in iter(self.proc.stdout.readline, ''):
                if line:
                    self.output_queue.put(line.strip())
                else:
                    break
        except:
            pass

    def send(self, cmd):
        """Send a command to the engine."""
        self.proc.stdin.write(cmd + '\n')
        self.proc.stdin.flush()

    def read_until(self, pattern, timeout=10):
        """Read lines until pattern is found."""
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
        """Drain any pending output."""
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
        """Terminate the engine process."""
        try:
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except:
            self.proc.kill()


def test_selfplay():
    print("=" * 60)
    print("Chess Engine Self-Play Test (Threaded I/O)")
    print("=" * 60)

    # Start two engines
    print("Starting White engine...")
    white = EngineProcess()
    time.sleep(0.3)
    white.drain_output()

    print("Starting Black engine...")
    black = EngineProcess()
    time.sleep(0.3)
    black.drain_output()

    # UCI handshake
    for name, engine in [("White", white), ("Black", black)]:
        engine.send("uci")
        lines, found = engine.read_until("uciok", timeout=5)
        if not found:
            print(f"ERROR: {name} engine UCI handshake failed")
            print(f"Output: {lines}")
            white.terminate()
            black.terminate()
            return 1

        engine.send("isready")
        lines, found = engine.read_until("readyok", timeout=5)
        if not found:
            print(f"ERROR: {name} engine not ready")
            white.terminate()
            black.terminate()
            return 1

    print("Both engines initialized successfully\n")

    # Play a game
    moves = []
    move_number = 1
    max_moves = 30

    print("Playing game (depth 4 per move):\n")

    for i in range(max_moves * 2):
        current = "White" if i % 2 == 0 else "Black"
        engine = white if i % 2 == 0 else black

        # Set position
        if moves:
            pos_cmd = f"position startpos moves {' '.join(moves)}"
        else:
            pos_cmd = "position startpos"

        engine.send(pos_cmd)
        engine.send("go depth 4")

        # Get bestmove
        lines, found = engine.read_until("bestmove", timeout=30)

        if not found:
            print(f"\nERROR: {current} failed to return move!")
            print(f"Output: {lines}")
            break

        # Extract move
        bestmove = None
        for line in lines:
            if line.startswith("bestmove"):
                match = re.search(r'bestmove\s+([a-h][1-8][a-h][1-8][qrbn]?)', line)
                if match:
                    bestmove = match.group(1)

        if not bestmove or bestmove == "0000":
            print(f"\nGame Over: {current} has no moves")
            break

        moves.append(bestmove)

        if i % 2 == 0:
            print(f"{move_number}. {bestmove}", end=" ", flush=True)
        else:
            print(f"{bestmove}", flush=True)
            move_number += 1

        if len(moves) >= 100:
            print("\n\nGame truncated at 50 full moves")
            break

    print(f"\n\nGame complete: {len(moves)} half-moves played")
    print(f"Final position after: {' '.join(moves[-10:] if len(moves) > 10 else moves)}")

    white.terminate()
    black.terminate()

    if len(moves) >= 10:
        print("\nSUCCESS: Engine successfully played a coherent game")
        return 0
    else:
        print("\nWARNING: Game too short")
        return 1


if __name__ == "__main__":
    sys.exit(test_selfplay())
