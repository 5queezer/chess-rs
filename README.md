# Chess Engine

A chess engine written in Rust supporting UCI and XBoard protocols.

## Features

- UCI and XBoard protocol support
- Alpha-beta search with aspiration windows
- Quiescence search
- Transposition table
- Killer moves and history heuristic
- Optional neural network evaluation
- 5 difficulty levels
- GPU acceleration support (CUDA/Metal)

## Building

```bash
cargo build --release
```

## Running

```bash
./target/release/chess
```

## UCI Commands

- `uci` - Initialize UCI mode
- `isready` - Check readiness
- `ucinewgame` - Start new game
- `position [fen <fen> | startpos] moves <moves>` - Set position
- `go [wtime <ms>] [btime <ms>] [winc <ms>] [binc <ms>] [movetime <ms>] [depth <d>]` - Search
- `stop` - Stop search
- `setoption name <name> value <value>` - Set option
- `perft <depth>` - Performance test

## XBoard Commands

- `xboard` - Initialize XBoard mode
- `protover 2` - Protocol version
- `new` - New game
- `force` - Force mode
- `go` - Start search
- `move <move>` - Make move
- `setboard <fen>` - Set position
- `level <mps> <base> <inc>` - Set time control
- `st <seconds>` - Time per move
- `sd <depth>` - Search depth
- `time <centiseconds>` - Engine time
- `otim <centiseconds>` - Opponent time
- `ping <n>` - Ping/pong
- `hint` - Get hint
- `undo` - Undo move
- `remove` - Undo two moves
- `quit` - Exit

## Options

- `Skill Level` (1-5): Playing strength
- `Use ML Evaluation` (true/false): Neural network evaluation
- `ML Model Path` (string): Model file path

## Move Format

Coordinate notation: `e2e4`, `e7e8q` (promotion)

## License

MIT
