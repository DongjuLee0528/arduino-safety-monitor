#!/usr/bin/env bash
# Experiment orchestration: Exp-A (EfficientNet-B0 lr=0.0003) and
#                           Exp-B (MobileNetV3-Large lr=0.001)
# Runs sequentially; never two MPS jobs concurrently.
# Safe to run in background: nohup bash mpu/ai/scripts/orchestrate_experiments.sh &
#
# Usage:
#   nohup bash mpu/ai/scripts/orchestrate_experiments.sh > mpu/ai/runs/orchestrate.log 2>&1 &
#   echo $! > mpu/ai/runs/orchestrate.pid
#
# Stop safely:
#   kill $(cat mpu/ai/runs/orchestrate.pid)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../../" && pwd)"
cd "$PROJECT_ROOT"
source .venv/bin/activate

LOG_DIR="$PROJECT_ROOT/mpu/ai/runs"
DATASET="/Users/dongjulee/AIdatasets/ helmet-safety-robot/raw/9rcv8mm682-4/Safety Helmet Wearing Dataset"

echo "======================================================"
echo " Experiment Orchestration"
echo " Start: $(date)"
echo " PID: $$"
echo "======================================================"

# ── verify production files are untouched before we start ─────────────────────
PROD_PTH="$PROJECT_ROOT/mpu/ai/models/best_model.pth"
PROD_ONNX="$PROJECT_ROOT/mpu/ai/models/best_model.onnx"
PROD_PTH_SHA=$(shasum -a 256 "$PROD_PTH" 2>/dev/null | awk '{print $1}')
PROD_ONNX_SHA=$(shasum -a 256 "$PROD_ONNX" 2>/dev/null | awk '{print $1}')
echo "[preflight] prod pth sha256:  $PROD_PTH_SHA"
echo "[preflight] prod onnx sha256: $PROD_ONNX_SHA"

# ── train helper — uses array to avoid word-splitting on spaces in PROJECT_ROOT ─
RUNS_DIR="$PROJECT_ROOT/mpu/ai/runs"

run_train() {
    # run_train ARCH LR EPOCHS LOG_FILE
    python3 -m mpu.ai.train \
        --architecture "$1" \
        --lr "$2" \
        --epochs "$3" \
        --batch-size 32 \
        --seed 42 \
        --patience 5 \
        --train-ratio 0.8 \
        --output-dir "$RUNS_DIR" \
        2>&1 | tee "$4"
}

run_dir_from_log() {
    awk -F'Run directory: ' '/Run directory:/ {d=$2} END {print d}' "$1"
}

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A — EfficientNet-B0 lr=0.0003
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "══════════════════════════════════════════════"
echo " EXP-A SMOKE TEST: EfficientNet-B0 lr=0.0003"
echo "══════════════════════════════════════════════"

run_train efficientnet_b0 0.0003 2 "$LOG_DIR/smoke_a.log"

EXP_A_SMOKE_DIR=$(run_dir_from_log "$LOG_DIR/smoke_a.log")
echo "[smoke-a] run dir: $EXP_A_SMOKE_DIR"

if [ ! -f "$EXP_A_SMOKE_DIR/best_metrics.json" ]; then
    echo "[FAIL] Exp-A smoke: best_metrics.json not created. Stopping."
    exit 1
fi
SMOKE_A_ACC=$(python3 -c "
import json
m = json.load(open('$EXP_A_SMOKE_DIR/best_metrics.json'))
print(m.get('best_val_accuracy', 'missing'))
")
echo "[smoke-a] val_accuracy=$SMOKE_A_ACC — PASS, proceeding to full training"

# ── Exp-A full training ───────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo " EXP-A FULL TRAINING: EfficientNet-B0 lr=0.0003 (max 50 epochs, patience 5)"
echo "══════════════════════════════════════════════"

run_train efficientnet_b0 0.0003 50 "$LOG_DIR/train_a.log"

EXP_A_DIR=$(run_dir_from_log "$LOG_DIR/train_a.log")
echo "[exp-a] run dir: $EXP_A_DIR"

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — MobileNetV3-Large lr=0.001
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "══════════════════════════════════════════════"
echo " EXP-B SMOKE TEST: MobileNetV3-Large lr=0.001"
echo "══════════════════════════════════════════════"

run_train mobilenet_v3_large 0.001 2 "$LOG_DIR/smoke_b.log"

EXP_B_SMOKE_DIR=$(run_dir_from_log "$LOG_DIR/smoke_b.log")
echo "[smoke-b] run dir: $EXP_B_SMOKE_DIR"

if [ ! -f "$EXP_B_SMOKE_DIR/best_metrics.json" ]; then
    echo "[FAIL] Exp-B smoke: best_metrics.json not created. Stopping."
    exit 1
fi
SMOKE_B_ACC=$(python3 -c "
import json
m = json.load(open('$EXP_B_SMOKE_DIR/best_metrics.json'))
print(m.get('best_val_accuracy', 'missing'))
")
echo "[smoke-b] val_accuracy=$SMOKE_B_ACC — PASS, proceeding to full training"

# ── Exp-B full training ───────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo " EXP-B FULL TRAINING: MobileNetV3-Large lr=0.001 (max 50 epochs, patience 5)"
echo "══════════════════════════════════════════════"

run_train mobilenet_v3_large 0.001 50 "$LOG_DIR/train_b.log"

EXP_B_DIR=$(run_dir_from_log "$LOG_DIR/train_b.log")
echo "[exp-b] run dir: $EXP_B_DIR"

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION — both experiments + ONNX + threshold analysis
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "══════════════════════════════════════════════"
echo " EVALUATION + ONNX + THRESHOLD ANALYSIS"
echo "══════════════════════════════════════════════"

python3 -m mpu.ai.scripts.evaluate_experiments \
    --exp-a-dir "$EXP_A_DIR" \
    --exp-b-dir "$EXP_B_DIR" \
    2>&1 | tee "$LOG_DIR/evaluate.log"

# ── verify production files still untouched ────────────────────────────────────
PROD_PTH_SHA2=$(shasum -a 256 "$PROD_PTH" 2>/dev/null | awk '{print $1}')
PROD_ONNX_SHA2=$(shasum -a 256 "$PROD_ONNX" 2>/dev/null | awk '{print $1}')
if [ "$PROD_PTH_SHA" != "$PROD_PTH_SHA2" ] || [ "$PROD_ONNX_SHA" != "$PROD_ONNX_SHA2" ]; then
    echo "[CRITICAL] Production files SHA256 changed! Training must not overwrite production."
    exit 1
fi
echo "[guard] Production files SHA256 unchanged — OK"

echo ""
echo "======================================================"
echo " Orchestration complete: $(date)"
echo "======================================================"
