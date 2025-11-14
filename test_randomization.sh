#!/bin/bash

# Test script to verify move randomization at low difficulty levels

ENGINE="./target/release/chess"

echo "Testing Move Randomization at Beginner Level"
echo "============================================="
echo ""
echo "Running the same position 10 times to verify randomization:"
echo ""

for i in {1..10}; do
    result=$(
        {
            echo "uci"
            sleep 0.05
            echo "setoption name Skill Level value 1"
            sleep 0.05
            echo "isready"
            sleep 0.05
            echo "position startpos"
            echo "go movetime 300"
            sleep 0.4
            echo "quit"
        } | $ENGINE 2>&1 | grep "bestmove" | head -1
    )
    echo "Run $i: $result"
done

echo ""
echo "If randomization is working, you should see different moves above."
echo ""
echo "Testing Easy Level (Level 2) Randomization:"
echo "-------------------------------------------"

for i in {1..10}; do
    result=$(
        {
            echo "uci"
            sleep 0.05
            echo "setoption name Skill Level value 2"
            sleep 0.05
            echo "isready"
            sleep 0.05
            echo "position startpos"
            echo "go movetime 300"
            sleep 0.4
            echo "quit"
        } | $ENGINE 2>&1 | grep "bestmove" | head -1
    )
    echo "Run $i: $result"
done

echo ""
echo "Testing Medium Level (Level 3) - Should be consistent:"
echo "------------------------------------------------------"

for i in {1..5}; do
    result=$(
        {
            echo "uci"
            sleep 0.05
            echo "setoption name Skill Level value 3"
            sleep 0.05
            echo "isready"
            sleep 0.05
            echo "position startpos"
            echo "go movetime 300"
            sleep 0.4
            echo "quit"
        } | $ENGINE 2>&1 | grep "bestmove" | head -1
    )
    echo "Run $i: $result"
done

echo ""
echo "Medium level should show the same move consistently (no randomization)."
