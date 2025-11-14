#!/bin/bash

# Test script for difficulty levels

ENGINE="./target/release/chess"

echo "Testing Chess Engine Difficulty Levels"
echo "========================================"
echo ""

# Function to test a difficulty level
test_difficulty() {
    local level=$1
    local name=$2

    echo "Testing Difficulty Level $level ($name)"
    echo "----------------------------------------"

    {
        echo "uci"
        sleep 0.1
        echo "setoption name Skill Level value $level"
        sleep 0.1
        echo "isready"
        sleep 0.1
        echo "position startpos"
        echo "go depth 10"
        sleep 2
        echo "quit"
    } | $ENGINE 2>&1 | grep -E "(uciok|readyok|Difficulty|bestmove|info depth)"

    echo ""
}

# Test each difficulty level
test_difficulty 1 "Beginner"
test_difficulty 2 "Easy"
test_difficulty 3 "Medium"
test_difficulty 4 "Hard"
test_difficulty 5 "Expert"

echo "Testing UCI Option Advertisement"
echo "--------------------------------"
{
    echo "uci"
    sleep 0.1
    echo "quit"
} | $ENGINE 2>&1 | grep -E "(option name|uciok)"

echo ""
echo "Testing Multiple Moves at Beginner Level"
echo "----------------------------------------"
{
    echo "uci"
    sleep 0.1
    echo "setoption name Skill Level value 1"
    sleep 0.1
    echo "isready"
    sleep 0.1
    echo "position startpos"
    echo "go movetime 500"
    sleep 0.7
    echo "position startpos moves e2e4"
    echo "go movetime 500"
    sleep 0.7
    echo "position startpos moves e2e4 e7e5"
    echo "go movetime 500"
    sleep 0.7
    echo "quit"
} | $ENGINE 2>&1 | grep -E "(bestmove|Difficulty)"

echo ""
echo "All tests completed!"
