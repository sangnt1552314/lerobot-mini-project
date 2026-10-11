#!/usr/bin/env bash
# End-to-end smoke test: check env -> 20 demos -> inspect -> short ACT run -> sim eval.
#
#   bash sim/run_all.sh
#
# Uses a timestamped dataset name and output dir so it never touches your main dataset or
# checkpoints. Stops at the first failing step. SIMULATION ONLY.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_env

STAMP="$(date +%Y%m%d_%H%M%S)"
export SIM_DATASET_NAME="${SIM_DATASET_NAME}_smoke_${STAMP}"
export OUTPUT_DIR="outputs/act_liftcube_smoke_${STAMP}"
SMOKE_DEMOS="${SMOKE_DEMOS:-20}"

info "Smoke run $STAMP: dataset $HF_USER/$SIM_DATASET_NAME, output $OUTPUT_DIR"

info "[1/4] Validating environment"
bash "$SIM_DIR/check.sh"

info "[2/4] Generating $SMOKE_DEMOS successful demos (+ dataset inspection)"
bash "$SIM_DIR/generate.sh" "$SMOKE_DEMOS"

info "[3/4] Training ACT for $SMOKE_TRAIN_STEPS steps"
bash "$SIM_DIR/train_act.sh" "$SMOKE_TRAIN_STEPS"

info "[4/4] Evaluating ACT in simulation"
bash "$SIM_DIR/eval_act.sh" "$OUTPUT_DIR/checkpoints/last/pretrained_model" "$EVAL_EPISODES"

info "Smoke pipeline finished. A short run on 20 demos is a plumbing test; expect a low success rate."
