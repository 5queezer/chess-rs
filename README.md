# Rust Chess Engine

This is a simple chess engine written in Rust. It communicates using the Universal Chess Interface (UCI) protocol.

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

The engine will start and wait for UCI commands to be sent to standard input.

## UCI Commands

The engine supports the following UCI commands:

*   `uci`: Tells the engine to use the UCI protocol. The engine will respond with its identity and a list of supported options.
*   `isready`: Asks the engine if it is ready to receive commands. The engine will respond with `readyok`.
*   `ucinewgame`: Tells the engine that the next search will be from a new game.
*   `position [fen <fenstring> | startpos] moves <move1> ... <moveN>`: Sets the board position. You can either specify a FEN string or start from the standard starting position. You can also specify a series of moves to be made from the given position.
*   `go`: Starts the search for the best move. The engine supports the following `go` command parameters:
    *   `wtime <ms>`: The amount of time the white player has left in milliseconds.
    *   `btime <ms>`: The amount of time the black player has left in milliseconds.
    *   `movetime <ms>`: The amount of time to search for a move in milliseconds.
    *   `depth <depth>`: The maximum depth to search to.
*   `stop`: Stops the search.
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
