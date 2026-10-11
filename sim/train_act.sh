#!/usr/bin/env bash
# Train LeRobot ACT on the simulated dataset. ACT reads state/action/camera shapes from it.
#
#   bash sim/train_act.sh              # TRAIN_STEPS from sim/.env (default 20000)
#   bash sim/train_act.sh 10000        # smoke run
#   RESUME=1 bash sim/train_act.sh     # continue the run in OUTPUT_DIR
#   OUTPUT_DIR=outputs/act_try2 bash sim/train_act.sh
#
# Extra arguments after STEPS go straight to lerobot-train. SIMULATION ONLY.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
load_env

STEPS="${1:-$TRAIN_STEPS}"
require_int "steps" "$STEPS"
shift $(( $# > 0 ? 1 : 0 ))
activate_env

[ -f "$DATASET_ROOT/meta/info.json" ] || die "No dataset at $DATASET_ROOT. Run: bash sim/generate.sh 20"

WANDB_ARGS=(--wandb.enable="$WANDB_ENABLE")
if [ "$WANDB_ENABLE" = "true" ]; then
    WANDB_ARGS+=(--wandb.project="$WANDB_PROJECT")
    if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
        warn "Not logged in to Weights & Biases; logging offline (sync later with 'wandb sync'). Run 'wandb login' to log online."
        WANDB_ARGS+=(--wandb.mode=offline)
    fi
fi

if [ -e "$OUTPUT_DIR" ]; then
    [ "${RESUME:-0}" = "1" ] || die "$OUTPUT_DIR already exists. Use RESUME=1 to continue it, or set OUTPUT_DIR to a new path."
    CONFIG="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json"
    [ -f "$CONFIG" ] || die "Cannot resume: $CONFIG not found."
    info "Resuming $OUTPUT_DIR up to $STEPS steps"
    lerobot-train --config_path="$CONFIG" --resume=true --steps="$STEPS" "${WANDB_ARGS[@]}" "$@"
else
    info "Training ACT on $DATASET_REPO_ID for $STEPS steps (batch $BATCH_SIZE, device $DEVICE)"
    lerobot-train \
        --dataset.repo_id="$DATASET_REPO_ID" \
        --dataset.root="$DATASET_ROOT" \
        --policy.type=act \
        --policy.device="$DEVICE" \
        --policy.push_to_hub=false \
        --output_dir="$OUTPUT_DIR" \
        --job_name=act_so101_liftcube_sim \
        --batch_size="$BATCH_SIZE" \
        --steps="$STEPS" \
        --save_freq="$SAVE_FREQ" \
        "${WANDB_ARGS[@]}" \
        "$@"
fi

CKPT="$OUTPUT_DIR/checkpoints/last/pretrained_model"
[ -f "$CKPT/config.json" ] || die "Training finished but $CKPT/config.json is missing."
info "Checkpoint: $CKPT"
info "Next: bash sim/eval_act.sh ${CKPT#"$REPO_ROOT"/}"
