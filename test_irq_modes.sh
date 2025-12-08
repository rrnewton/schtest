#!/bin/bash
# Test each IRQ disruption mode individually to see interrupt impact
# Usage: ./test_irq_modes.sh [duration]

DURATION=${1:-20}
TEST_BINARY="./target/release/schtest"

if [ ! -f "$TEST_BINARY" ]; then
    echo "Error: $TEST_BINARY not found. Please build first."
    exit 1
fi

# Modes to test
MODES=("none" "futex" "pmu" "timer" "combined")

echo "Testing IRQ modes with $DURATION second duration"
echo "================================================"
echo ""

for mode in "${MODES[@]}"; do
    echo ""
    echo "========================================="
    echo "Testing mode: $mode"
    echo "========================================="

    # Start interrupt monitor in background
    monitor_log="/tmp/irq_monitor_${mode}.log"
    ./monitor_interrupts.sh 1 "0 1 2" > "$monitor_log" 2>&1 &
    monitor_pid=$!

    echo "Monitor started (PID: $monitor_pid), waiting 2s for baseline..."
    sleep 2

    echo "Starting test with mode=$mode..."
    SCHTEST_IRQ_MODE="$mode" SCHTEST_IRQ_DURATION="$DURATION" "$TEST_BINARY" --filter irq --nocapture 2>&1 | grep -E "(WARNING|mode:|CPU [0-9]|Interrupt Summary|Victim CPU)"

    echo "Test completed, collecting final stats..."
    sleep 2

    # Stop monitor
    kill $monitor_pid 2>/dev/null
    wait $monitor_pid 2>/dev/null

    # Show interrupt summary (last 5 samples)
    echo ""
    echo "Interrupt time summary for $mode (last 5 samples):"
    tail -20 "$monitor_log" | head -18

    echo ""
    echo "Press Enter to continue to next mode..."
    read
done

echo ""
echo "All tests completed. Monitor logs saved to /tmp/irq_monitor_*.log"
