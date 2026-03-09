#!/bin/bash
# =============================================================================
# MCP-Bench Data Collection Pipeline
#
# Complete workflow: Collect → Filter → Convert → Ready for Training
#
# Usage:
#   # Step 1: Collect (pick one)
#   bash pipeline.sh collect-single <task_id> [tasks_file]
#   bash pipeline.sh collect-batch [tasks_file] [max_tasks]
#
#   # Step 2: Filter
#   bash pipeline.sh filter              # Rule-based only (free)
#   bash pipeline.sh filter-full         # Rule-based + LLM judge (costs tokens)
#
#   # Step 3: Convert to SFT format
#   bash pipeline.sh convert
#
#   # Or run everything:
#   bash pipeline.sh all [tasks_file] [max_tasks]
# =============================================================================

set -e
export PATH="$HOME/.local/bin:$PATH"

TRAJ_DIR="trajectories_openclaw"
TRAIN_DIR="training_data"
TASKS_FILE="${2:-tasks/mcpbench_tasks_single_runner_format.json}"
MAX_TASKS="${3:-5}"
MAX_ROUNDS=30

case "${1}" in

  collect-single)
    TASK_ID="${2:?Usage: pipeline.sh collect-single <task_id> [tasks_file]}"
    TASKS_FILE="${3:-tasks/mcpbench_tasks_single_runner_format.json}"
    echo "=== Collecting single task: $TASK_ID ==="
    python3 run_openclaw_benchmark.py \
      --task-id "$TASK_ID" \
      --tasks-file "$TASKS_FILE" \
      --trajectory-dir "$TRAJ_DIR" \
      --max-rounds $MAX_ROUNDS \
      --no-eval
    ;;

  collect-batch)
    echo "=== Batch collecting from $TASKS_FILE (max $MAX_TASKS tasks) ==="
    python3 run_openclaw_benchmark.py \
      --tasks-file "$TASKS_FILE" \
      --trajectory-dir "$TRAJ_DIR" \
      --max-rounds $MAX_ROUNDS \
      --max-tasks "$MAX_TASKS" \
      --no-eval
    ;;

  filter)
    echo "=== Stage 1: Rule-based filtering ==="
    python3 tools/filter_trajectories.py \
      --input-dir "$TRAJ_DIR" \
      --output-dir "$TRAIN_DIR" \
      --rules-only
    ;;

  filter-full)
    echo "=== Stage 1 + 2: Rule-based + LLM Judge filtering ==="
    python3 tools/filter_trajectories.py \
      --input-dir "$TRAJ_DIR" \
      --output-dir "$TRAIN_DIR"
    ;;

  convert)
    echo "=== Converting to SFT format ==="
    python3 tools/convert_to_sft.py \
      --input-dir "$TRAJ_DIR" \
      --output "$TRAIN_DIR/sft_data.jsonl" \
      --reasoning strip
    echo ""
    echo "Output files:"
    echo "  $TRAIN_DIR/sft_data.jsonl                (standard OpenAI format)"
    echo "  $TRAIN_DIR/sft_data_with_reasoning.jsonl  (with <think> blocks)"
    ;;

  all)
    echo "=== Full Pipeline: Collect → Filter → Convert ==="
    echo ""
    $0 collect-batch "$TASKS_FILE" "$MAX_TASKS"
    echo ""
    $0 filter
    echo ""
    $0 convert
    echo ""
    echo "=== Pipeline complete ==="
    echo "Training data ready at: $TRAIN_DIR/"
    ;;

  *)
    echo "MCP-Bench Data Collection Pipeline"
    echo ""
    echo "Usage:"
    echo "  bash pipeline.sh collect-single <task_id> [tasks_file]"
    echo "  bash pipeline.sh collect-batch [tasks_file] [max_tasks]"
    echo "  bash pipeline.sh filter              # Rule-based only (free)"
    echo "  bash pipeline.sh filter-full         # + LLM judge (costs tokens)"
    echo "  bash pipeline.sh convert             # To OpenAI SFT format"
    echo "  bash pipeline.sh all [tasks_file] [max_tasks]  # Everything"
    echo ""
    echo "Environment variables needed:"
    echo "  KIMI_API_KEY    - Required for collection and LLM judge"
    echo "  KIMI_BASE_URL   - Default: https://api.moonshot.cn/v1"
    ;;

esac
