#!/bin/bash
# Parallel lesson generation with Claude Sonnet 4.5
#
# Usage:
#   ./generate_batch_lessons.sh [LESSONS] [CATEGORY] [WORKERS]
#
# Examples:
#   ./generate_batch_lessons.sh 500              # 500 random lessons, 5 workers
#   ./generate_batch_lessons.sh 100 a1          # 100 A1 (beginner) lessons, 5 workers
#   ./generate_batch_lessons.sh 200 b2 10       # 200 B2 (security) lessons, 10 workers
#   ./generate_batch_lessons.sh 50 refactoring  # 50 refactoring lessons, 5 workers

set -e

# Parse command-line arguments
TOTAL_LESSONS=${1:-500}
CATEGORY=${2:-}  # Optional category filter (e.g., a1, b2, refactoring)
WORKERS=${3:-5}  # Number of parallel workers

PROVIDER="anthropic:claude-sonnet-4-5-20250929"

# Load .env file if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

echo "🚀 Parallel Lesson Generation"
echo "================================"
echo "   Total lessons: $TOTAL_LESSONS"
echo "   Category: ${CATEGORY:-ALL (random mix)}"
echo "   Workers: $WORKERS"
echo "   Provider: $PROVIDER"
echo ""

# Check API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ ERROR: ANTHROPIC_API_KEY not set"
    echo ""
    echo "Please set your API key:"
    echo "  export ANTHROPIC_API_KEY='sk-ant-...'"
    echo ""
    echo "Get your key at: https://console.anthropic.com/settings/keys"
    exit 1
fi

echo "✓ API key found: ${ANTHROPIC_API_KEY:0:20}..."
echo ""

# Function to run learning cycles for one worker
run_worker() {
    worker_id=$1
    cycles=$2
    category=$3

    echo "[Worker $worker_id] Starting $cycles cycles"

    for i in $(seq 1 $cycles); do
        echo "[Worker $worker_id] Cycle $i/$cycles"

        # Build command with optional category filter
        cmd="python -m nerion_digital_physicist.learning_orchestrator --provider \"$PROVIDER\""
        if [ -n "$category" ]; then
            cmd="$cmd --category \"$category\""
        fi

        # Run one learning cycle
        eval $cmd 2>&1 | tee -a "logs/worker_${worker_id}.log" | \
            grep -E "(Starting|Success|Failed|ERROR|Self-vetting|saved)" || true

        # Brief pause to avoid rate limits (2 seconds between cycles)
        sleep 2

        # Every 10 cycles, show progress
        if [ $((i % 10)) -eq 0 ]; then
            count=$(sqlite3 out/learning/curriculum.sqlite "SELECT COUNT(*) FROM lessons;" 2>/dev/null || echo "0")
            echo "[Worker $worker_id] Checkpoint: $count total lessons in database"
        fi
    done

    echo "[Worker $worker_id] Complete!"
}

# Create logs directory
mkdir -p logs

# Calculate cycles per worker
cycles_per_worker=$((TOTAL_LESSONS / WORKERS))
echo "📋 Each worker will run $cycles_per_worker cycles"
echo ""

# Get starting lesson count
start_count=$(sqlite3 out/learning/curriculum.sqlite "SELECT COUNT(*) FROM lessons;" 2>/dev/null || echo "0")
echo "📊 Starting with $start_count lessons"
echo ""

# Record start time
start_time=$(date +%s)

echo "🏁 Launching $WORKERS workers..."
echo ""

# Launch workers in parallel (background processes)
for worker in $(seq 1 $WORKERS); do
    run_worker $worker $cycles_per_worker "$CATEGORY" &
    pids[${worker}]=$!
done

echo "✓ All workers launched"
echo ""
echo "Workers running in background. To monitor progress:"
echo "  tail -f logs/worker_1.log"
echo "  sqlite3 out/learning/curriculum.sqlite 'SELECT COUNT(*) FROM lessons;'"
echo ""

# Wait for all workers to complete
for worker in $(seq 1 $WORKERS); do
    wait ${pids[$worker]}
    echo "✓ Worker $worker finished"
done

# Record end time
end_time=$(date +%s)
elapsed=$((end_time - start_time))
elapsed_mins=$((elapsed / 60))

# Get final count
final_count=$(sqlite3 out/learning/curriculum.sqlite "SELECT COUNT(*) FROM lessons;" 2>/dev/null || echo "0")
generated=$((final_count - start_count))

echo ""
echo "================================"
echo "✅ BATCH GENERATION COMPLETE"
echo "================================"
echo "   Started with: $start_count lessons"
echo "   Generated: $generated new lessons"
echo "   Final total: $final_count lessons"
echo "   Time elapsed: ${elapsed_mins} minutes"
if [ $((cycles_per_worker * WORKERS)) -gt 0 ]; then
    echo "   Success rate: $((generated * 100 / (cycles_per_worker * WORKERS)))%"
fi
echo ""
echo "📦 Database: out/learning/curriculum.sqlite"
echo ""
