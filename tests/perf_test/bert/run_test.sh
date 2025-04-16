#!/bin/bash

# Function to clean up server process
cleanup() {
    pkill -f "python tests/perf_test/bert/server.py"
}

# Trap script exit to run cleanup
trap cleanup EXIT

# Get the mode argument, default to 'default'
MODE=${1:-default}
echo "Testing LitServe in $MODE mode"

# Start the server in the background and capture its PID
python tests/perf_test/bert/server.py --mode $MODE &
SERVER_PID=$!

echo "Server started with PID $SERVER_PID in $MODE mode"

# Run your benchmark script
echo "Preparing to run benchmark.py..."

export PYTHONPATH=$PWD && python tests/perf_test/bert/benchmark.py

# Check if benchmark.py exited successfully
if [ $? -ne 0 ]; then
    echo "benchmark.py failed to run successfully."
    exit 1
else
    echo "benchmark.py ran successfully."
fi
