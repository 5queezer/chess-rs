# Rust Chess Engine with ML Capabilities

A chess engine written in Rust with support for both UCI and XBoard/WinBoard protocols. Features include classical evaluation, optional AlphaZero-style neural network evaluation, difficulty levels, and full compatibility with popular chess GUIs.

## Building

To build the engine, you will need to have Rust and Cargo installed. You can then build the project by running the following command in the root directory of the project:

```
cargo build --release
```

This will create an optimized executable at `target/release/chess`.

## Running

You can run the engine directly with Cargo:

```
cargo run
```

Alternatively, you can run the compiled executable:

```
./target/release/chess
```

The engine will start and wait for UCI or XBoard commands to be sent to standard input.

## Supported Protocols

The engine supports two chess protocols:

- **UCI (Universal Chess Interface)** - Modern protocol used by most chess GUIs
- **XBoard/WinBoard Protocol v2** - Classic protocol with full compatibility

### Compatible Chess GUIs

- **UCI GUIs**: Arena, Fritz, ChessBase, Cute Chess, Banksia GUI, Lucas Chess
- **XBoard GUIs**: XBoard, WinBoard, ChessX, Cute Chess, PyChess, Arena

## Features

- **Dual Protocol Support**: Seamlessly works with both UCI and XBoard interfaces
- **Advanced Search**: Alpha-beta pruning with aspiration windows, null move pruning, killer moves, and history heuristic
- **ML Evaluation**: Optional AlphaZero-style ResNet neural network evaluation with GPU support (CUDA/Metal)
- **Classical Evaluation**: Hand-crafted evaluation with piece-square tables, pawn structure, and positional bonuses
- **Difficulty Levels**: 5 skill levels from Beginner to Expert
- **Time Management**: Intelligent time allocation for various time controls
- **Opening Book**: None (uses search from move 1)
- **Endgame Tablebases**: Not supported

## UCI Commands

The engine supports the following UCI commands:

*   `uci`: Tells the engine to use the UCI protocol. The engine will respond with its identity and a list of supported options.
*   `isready`: Asks the engine if it is ready to receive commands. The engine will respond with `readyok`.
*   `ucinewgame`: Tells the engine that the next search will be from a new game.
*   `position [fen <fenstring> | startpos] moves <move1> ... <moveN>`: Sets the board position. You can either specify a FEN string or start from the standard starting position. You can also specify a series of moves to be made from the given position.
*   `go`: Starts the search for the best move. The engine supports the following `go` command parameters:
    *   `wtime <ms>`: The amount of time the white player has left in milliseconds.
    *   `btime <ms>`: The amount of time the black player has left in milliseconds.
    *   `winc <ms>`: White's time increment per move in milliseconds.
    *   `binc <ms>`: Black's time increment per move in milliseconds.
    *   `movetime <ms>`: The amount of time to search for a move in milliseconds.
    *   `depth <depth>`: The maximum depth to search to.
    *   `movestogo <n>`: Number of moves to next time control.
*   `stop`: Stops the search.
*   `setoption name <name> value <value>`: Sets an engine option (Skill Level, Use ML Evaluation, ML Model Path).
*   `perft <depth>`: Runs a performance test to the given depth. This is useful for debugging the move generation.

### Example

Here is an example of how to use the engine to find the best move from the starting position with a 1-second search time:

```
position startpos
go movetime 1000
```

The engine will respond with the best move it found, for example:

```
bestmove e2e4
```

## XBoard/WinBoard Commands

The engine fully supports the XBoard/WinBoard protocol version 2. This makes it compatible with ChessX, XBoard, WinBoard, and other classic chess interfaces.

### Core Commands

*   `xboard`: Tells the engine to use the XBoard protocol.
*   `protover 2`: Requests protocol version 2. The engine will respond with its supported features.
*   `new`: Starts a new game. The engine will play Black by default.
*   `quit`: Exits the engine.

### Position Setup

*   `setboard <fen>`: Sets the board position using a FEN string.
*   `position startpos`: Sets the board to the starting position (also works in XBoard mode).

### Move Commands

*   `go`: Tells the engine to start thinking on the current position.
*   `move <move>`: Makes a move on the board (e.g., `move e2e4`). The engine will automatically respond if not in force mode.
*   `usermove <move>`: Alternative move format (if usermove feature is enabled).
*   `force`: Enters force mode - the engine will not think or move automatically.
*   `playother`: Leaves force mode and tells the engine to play the opposite color.

### Time Controls

*   `level <mps> <base> <inc>`: Sets time controls
    *   `mps`: Moves per session (0 for game in base time)
    *   `base`: Base time in minutes
    *   `inc`: Increment in seconds
    *   Example: `level 40 5 0` = 40 moves in 5 minutes
    *   Example: `level 0 3 2` = 3 minutes + 2 second increment per move
*   `st <time>`: Sets a fixed time limit of `time` seconds per move.
*   `sd <depth>`: Sets a fixed search depth limit.
*   `time <n>`: Sets the engine's remaining time in centiseconds (1/100th of a second).
*   `otim <n>`: Sets the opponent's remaining time in centiseconds.

### Thinking Output

*   `post`: Enables thinking output during search.
*   `nopost`: Disables thinking output.

When `post` is enabled, the engine outputs thinking information in the format:
```
<depth> <score> <time> <nodes> <pv>
```

Example:
```
5 25 123 45678 e2e4
```
(Depth 5, score +0.25, 1.23 seconds, 45,678 nodes, best move e2e4)

### Other Commands

*   `ping <n>`: Synchronization command. The engine will respond with `pong <n>`.
*   `?`: Tells the engine to move immediately (interrupt current search).
*   `hint`: Asks the engine for a hint about the best move.
*   `undo`: Takes back one move.
*   `remove`: Takes back two moves (one for each side).
*   `hard`: Enables pondering (not implemented - acknowledged but ignored).
*   `easy`: Disables pondering (not implemented - acknowledged but ignored).
*   `result <result>`: Informs the engine of the game result (e.g., `result 1-0`).
*   `draw`: Offers a draw to the engine (currently declined automatically).
*   `computer`: Informs the engine that the opponent is a computer.
*   `name <name>`: Informs the engine of the opponent's name.
*   `rating <engine_rating> <opponent_rating>`: Informs the engine of player ratings.

### Feature Negotiation

After receiving `protover 2`, the engine advertises these features:

*   `myname="rce-ml"`: Engine name
*   `variants="normal"`: Only supports standard chess
*   `setboard=1`: Supports FEN position setup
*   `usermove=0`: Does not require "usermove" prefix
*   `time=1`: Supports time controls
*   `draw=1`: Supports draw offers
*   `sigint=0`, `sigterm=0`: No signal handling
*   `reuse=1`: Can reuse position information
*   `analyze=0`: Does not support analysis mode
*   `ping=1`: Supports ping/pong synchronization
*   `playother=1`: Supports switching sides
*   `colors=0`: No color preference
*   `names=0`: Does not need opponent names (but accepts them)
*   `san=0`: Uses coordinate notation (e.g., e2e4), not SAN

### XBoard Examples

**Example 1: Basic game**
```
xboard
protover 2
new
e2e4
move e7e5
```

**Example 2: Time control setup**
```
xboard
protover 2
level 0 5 0
new
go
```
(Engine plays White with 5 minutes total)

**Example 3: Fixed time per move**
```
xboard
protover 2
st 10
new
go
```
(Engine uses 10 seconds per move)

**Example 4: Force mode analysis**
```
xboard
protover 2
new
force
e2e4
e7e5
hint
```
(Set up position without engine moving, then ask for a hint)

**Example 5: Position setup from FEN**
```
xboard
setboard rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
go
```

## Engine Options

The engine supports the following configurable options (available in both protocols):

*   **Skill Level** (1-5): Controls the playing strength
    *   1 = Beginner (depth 2, 40% randomization)
    *   2 = Easy (depth 3, 20% randomization)
    *   3 = Medium (depth 5)
    *   4 = Hard (depth 7)
    *   5 = Expert (depth 100, full strength)
*   **Use ML Evaluation** (true/false): Enables or disables neural network evaluation
*   **ML Model Path** (string): Path to custom neural network model file (optional)

## Using with Chess GUIs

### ChessX

1. Download or build the engine
2. Open ChessX and go to **Configure → Engines**
3. Click **Add** and select the compiled binary
4. Choose **WinBoard** as the protocol
5. The engine will appear in the engine list for games and analysis

### XBoard/WinBoard

Add the engine to your `~/.xboardrc` (Linux/Mac) or use the Engine menu:

```
-fcp "./target/release/chess"
-fd /path/to/chess-rs
```

Or via the GUI: **Engine → Edit Engine List**

### Arena

1. Go to **Engines → Install New Engine**
2. Select the engine binary
3. Choose **WinBoard** or **UCI** as the protocol
4. Configure time controls and options as desired

### Cute Chess

Create an engine configuration file or use the GUI:

```bash
cutechess-cli -engine name=rce-ml cmd=./target/release/chess proto=xboard
```

## Technical Details

### Search Algorithm

- **Iterative Deepening**: Searches incrementally deeper until time runs out
- **Aspiration Windows**: Narrows search window for faster results at higher depths
- **Alpha-Beta Pruning**: Efficient tree search with cutoffs
- **Null Move Pruning**: Reduces search tree by testing "passing" moves
- **Quiescence Search**: Extends search for tactical positions
- **Move Ordering**: Killer moves and history heuristic for better pruning
- **Transposition Table**: Caches position evaluations to avoid re-searching

### Evaluation

The engine uses a hybrid evaluation approach:

1. **Classical Evaluation** (always available):
   - Material counting (P=100, N=320, B=330, R=500, Q=900)
   - Piece-square tables for positional play
   - Bishop pair bonus (+40)
   - Rook on open file (+25) and semi-open file (+12)
   - Pawn structure penalties (doubled -12, isolated -15)

2. **Neural Network Evaluation** (optional, GPU-accelerated):
   - AlphaZero-style ResNet architecture
   - 20 residual blocks with batch normalization
   - Dual-head output (value + policy)
   - Automatic fallback to classical if ML unavailable
   - Supports CUDA and Metal acceleration

### Time Management

- **UCI Mode**: Allocates time based on `wtime`, `btime`, `winc`, `binc` parameters
- **XBoard Mode**: Uses `level`, `st`, `sd`, `time`, and `otim` commands
- **Default Allocation**: Approximately 1/40th of remaining time plus increment
- **Fixed Time**: Can be set via `st` (XBoard) or `movetime` (UCI)
- **Fixed Depth**: Can be set via `sd` (XBoard) or `depth` (UCI)

### Move Format

All moves use long algebraic notation (coordinate notation):
- Normal move: `e2e4`
- Capture: `e4d5` (no special notation)
- Castling: `e1g1` (kingside), `e1c1` (queenside)
- Promotion: `e7e8q` (queen), `e7e8n` (knight), etc.
- En passant: `e5d6` (no special notation)

## Performance Testing

The engine includes a `perft` command for verifying move generation correctness:

```
perft 5
```

This performs a performance test to depth 5, counting all legal positions.

## Contributing

See the additional documentation files for more information:
- `DIFFICULTY_LEVELS.md` - Detailed explanation of skill level system
- `ML_README.md` - Machine learning capabilities and setup

## License

This project is a chess engine implementation in Rust with educational purposes.
