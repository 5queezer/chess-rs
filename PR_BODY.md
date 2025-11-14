## Summary

This PR optimizes the chess engine's performance to ensure all moves complete within 60 seconds, addressing the issue where ML features were causing extremely slow move times.

## Changes

### 1. Hard Time Limit Enforcement (60 seconds max)
- Added `MAX_MOVE_TIME_MS = 60000` constant
- Enforces maximum time even when longer times are requested
- Prevents engine from taking too long on difficult positions

### 2. Adaptive Search Depth
Implemented intelligent depth limiting based on available time and evaluation method:

**With ML enabled:**
- < 1s: depth 3
- < 5s: depth 5
- < 15s: depth 7
- < 30s: depth 10
- ≥ 30s: depth 12 max

**Classical evaluation (default):**
- < 1s: depth 6
- < 5s: depth 10
- < 15s: depth 15
- ≥ 15s: depth 100 (search until timeout)

### 3. Smart Time Allocation
- **ML Time Multiplier (0.3×)**: Reduces allocation to 30% when ML is enabled (ML is ~10× slower)
- **Safety Margin (100ms)**: Reserves time to avoid timeout edge cases
- **Conservative defaults**: Uses 40 moves to go instead of 60

### 4. ML Disabled by Default
- Current ML implementation uses random (untrained) weights
- Not suitable for actual gameplay
- Can be enabled via: `setoption name Use ML Evaluation value true`
- Dramatically improves performance for regular play

### 5. Explicit Depth Request Handling
- When user specifies explicit depth (e.g., `go depth 20`), it's honored exactly
- Adaptive limits only apply when depth is not specified
- Fixes issue where `go depth 20` was capped at depth 15

## Performance Results

**Classical Evaluation (default):**
- ~500K-1M nodes/second
- Depth 10 in ~30 seconds
- Depth 20 in ~60 seconds with 26M+ nodes searched

**ML Evaluation (when enabled):**
- ~5K-30K nodes/second
- Depth 7 in ~10 seconds
- Adaptive depth capping prevents excessive slowdown

## Testing

All tests pass:
```
test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured
```

Verified functionality:
- ✓ Explicit depth requests honored (`go depth 20` → depth 20)
- ✓ Adaptive depth works (`go movetime 3000` → depth 10)
- ✓ Time limits enforced (max 60 seconds per move)
- ✓ ML can be toggled on/off via UCI option

## Impact

All moves now complete **within 60 seconds** regardless of configuration, providing a much more responsive gameplay experience while maintaining the option to use ML features when a trained model becomes available.
