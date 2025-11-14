#!/bin/bash

# Test script to verify depth limits at different difficulty levels

ENGINE="./target/release/chess"

echo "Testing Depth Limits at Different Difficulty Levels"
echo "===================================================="
echo ""

test_depth() {
    local level=$1
    local name=$2
    local expected_depth=$3

    echo "Level $level ($name) - Expected max depth: $expected_depth"

    result=$(
        {
            echo "uci"
            sleep 0.05
            echo "setoption name Skill Level value $level"
            sleep 0.05
            echo "isready"
            sleep 0.05
            echo "position startpos"
            echo "go depth 100"
            sleep 1
            echo "quit"
        } | $ENGINE 2>&1 | grep "info depth" | tail -1
    )

    echo "  Result: $result"
    echo ""
}

test_depth 1 "Beginner" 2
test_depth 2 "Easy" 3
test_depth 3 "Medium" 5
test_depth 4 "Hard" 7
test_depth 5 "Expert" "100+"

echo "Testing that Expert level respects explicit depth request:"
echo "----------------------------------------------------------"

result=$(
    {
        echo "uci"
        sleep 0.05
        echo "setoption name Skill Level value 5"
        sleep 0.05
        echo "isready"
        sleep 0.05
        echo "position startpos"
        echo "go depth 8"
        sleep 1
        echo "quit"
    } | $ENGINE 2>&1 | grep "info depth" | tail -1
)

echo "Expert level with 'go depth 8': $result"
echo ""

echo "Testing time-based search at different levels:"
echo "-----------------------------------------------"

for level in 1 2 3 4 5; do
    result=$(
        {
            echo "uci"
            sleep 0.05
            echo "setoption name Skill Level value $level"
            sleep 0.05
            echo "isready"
            sleep 0.05
            echo "position startpos"
            echo "go movetime 1000"
            sleep 1.2
            echo "quit"
        } | $ENGINE 2>&1 | grep "info depth" | tail -1
    )
    echo "Level $level (1000ms): $result"
done

echo ""
echo "All depth limit tests completed!"
