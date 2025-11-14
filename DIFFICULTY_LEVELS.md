# Chess Engine Difficulty Levels

This chess engine now supports 5 difficulty levels to accommodate players of different skill levels.

## Setting Difficulty Level

### UCI Protocol

```
setoption name Skill Level value <1-5>
```

Example:
```
uci
setoption name Skill Level value 2
position startpos
go movetime 5000
```

### Difficulty Levels

| Level | Name     | Search Depth | Randomization | Typical Nodes | Description |
|-------|----------|--------------|---------------|---------------|-------------|
| 1     | Beginner | 2            | 40%           | ~150-200      | Very weak play, makes frequent mistakes. Good for absolute beginners. |
| 2     | Easy     | 3            | 20%           | ~600-800      | Weak play with occasional mistakes. Suitable for novice players. |
| 3     | Medium   | 5            | None          | ~5,000        | Moderate strength. Good for intermediate players. |
| 4     | Hard     | 7            | None          | ~40,000-50,000| Strong play. Challenging for most players. |
| 5     | Expert   | 100 (default)| None          | 500,000+      | Maximum strength. Very strong play. |

## How It Works

### Search Depth Limiting

At lower difficulty levels, the engine searches to a limited depth:
- Beginner (1): 2 ply deep
- Easy (2): 3 ply deep
- Medium (3): 5 ply deep
- Hard (4): 7 ply deep
- Expert (5): Full depth (up to 100 ply or time limit)

### Move Randomization

At the two lowest difficulty levels, the engine adds randomization:

- **Beginner (40% chance)**: Selects a random move from the first 5 legal moves instead of the best move
- **Easy (20% chance)**: Selects a random move from the first 3 legal moves instead of the best move
- **Medium and above**: No randomization, always plays the best move found

This randomization makes lower levels more human-like and less predictable, preventing the same games from being played repeatedly.

## Testing

Three test scripts are included:

1. **test_difficulty.sh**: Tests all 5 difficulty levels and UCI option advertisement
2. **test_randomization.sh**: Verifies move randomization at lower levels
3. **test_depth_limits.sh**: Confirms depth limits are enforced correctly

Run any test with:
```bash
./test_difficulty.sh
./test_randomization.sh
./test_depth_limits.sh
```

## Examples

### Playing at Beginner Level
```
uci
setoption name Skill Level value 1
isready
position startpos
go movetime 1000
```

### Playing at Expert Level (Default)
```
uci
isready
position startpos
go movetime 5000
```

### Switching Difficulty Mid-Game
You can change the difficulty level at any time:
```
uci
position startpos
setoption name Skill Level value 1
go movetime 1000
position startpos moves e2e4
setoption name Skill Level value 5
go movetime 5000
```

## Performance Characteristics

| Level | Time for Move (1000ms limit) | Nodes Searched | ELO Estimate |
|-------|------------------------------|----------------|--------------|
| 1     | < 1ms                        | ~150           | ~800-1000    |
| 2     | < 5ms                        | ~600           | ~1200-1400   |
| 3     | < 50ms                       | ~5,000         | ~1600-1800   |
| 4     | < 200ms                      | ~45,000        | ~2000-2200   |
| 5     | Full time allocation         | 300,000+       | ~2400+       |

*Note: ELO estimates are approximate and may vary based on position complexity.*

## Implementation Details

The difficulty system modifies three aspects of the engine:

1. **Maximum search depth** - Lower levels stop searching earlier
2. **Move selection** - Lower levels occasionally pick suboptimal moves
3. **Time management** - All levels use the same time allocation strategy

The transposition table, killer moves, history heuristic, and other optimizations remain active at all levels.

## Compatibility

- Fully compatible with UCI protocol
- Works with both `go depth N` and `go movetime N` commands
- Expert level (5) respects explicit depth requests
- Lower levels override depth requests to enforce difficulty constraints
- Default level is 5 (Expert) for backward compatibility
