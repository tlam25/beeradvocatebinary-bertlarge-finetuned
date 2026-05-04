#!/bin/bash
# Launcher for train.py — runs in background with logging.
# Pass any train.py args through, e.g.:
#     ./run.sh                              # run all 5 seeds × 4 aspects
#     ./run.sh --seeds 2025 42              # only 2 seeds
#     ./run.sh --aspects appearance aroma   # only 2 aspects

set -e

# ---- HuggingFace cache (uncomment + edit if your $HOME is small) ----
# export HF_HOME=/path/to/large/disk/hf_cache

# ---- GPU selection ----
export CUDA_VISIBLE_DEVICES=0

mkdir -p logs results checkpoints

LOG="logs/run_$(date +%Y%m%d_%H%M%S).log"

nohup python -u train.py "$@" > "$LOG" 2>&1 &
PID=$!

echo "Started PID=$PID"
echo "Log file: $LOG"
echo ""
echo "Theo dõi log:    tail -f $LOG"
echo "Kiểm tra job:    ps -p $PID"
echo "Dừng job:        kill $PID"