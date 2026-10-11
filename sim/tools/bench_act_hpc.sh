#!/usr/bin/env bash
# Throughput benchmark for ACT training on one H200 (run inside a GPU job: pbs/test_act_h200.pbs).
# Short runs that change one thing at a time and keep what helps:
#   1 repo defaults (batch 8, 4 workers, torchcodec = LeRobot's default backend, fp32/TF32)
#   2 workers = NCPUS-2                     3 same with pyav       -> keep faster backend
#   4-6 batch 32 / 64 / 128                                         -> keep best samples/s
#   7 + bf16 autocast                       8 + dataset on /dev/shm
# Each run: BENCH_STEPS steps, log every 25, first 2 intervals dropped, no checkpoints, no W&B.
# Writes <bench dir>/<name>.{log,usage.csv,json}, nvidia-smi dmon output, and summary.md.
#
#   bash sim/tools/bench_act_hpc.sh [BENCH_DIR]
#
# Numbers from the SYNTHETIC dataset are provisional: real sim frames compress differently, and a
# 20-episode dataset makes the DataLoader restart its workers (an epoch boundary) more often.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRATCH="${SCRATCH_ROOT:-/scratch/e1583535}"
BENCH="$(realpath -m "${1:-$REPO_ROOT/logs/bench_act/${PBS_JOBID:-$(date +%Y%m%d_%H%M%S)}}")"
BENCH_STEPS="${BENCH_STEPS:-300}"
BASE_STEPS=20000; BASE_BATCH=8          # default recipe; full-run estimate = same number of epochs
export DATASET_REPO_ID="${DATASET_REPO_ID:-local/synthetic_liftcube_smoke}"
export DATASET_ROOT="${DATASET_ROOT:-$SCRATCH/datasets/lerobot/$DATASET_REPO_ID}"
export NCPUS="${NCPUS:-$(nproc)}"
export WANDB_ENABLE=false LOG_FREQ=25
MAXW=$(( NCPUS > 3 ? NCPUS - 2 : 1 ))
mkdir -p "$BENCH"
echo "bench dir $BENCH, dataset $DATASET_ROOT, NCPUS $NCPUS, $BENCH_STEPS steps per run"

nvidia-smi dmon -s um -d 2 -o T > "$BENCH/nvidia-smi-dmon.txt" 2>&1 &
DMON=$!
trap 'kill $DMON 2>/dev/null' EXIT

# run NAME BATCH WORKERS BACKEND PRECISION STAGE -> prints samples/s (0 on failure)
run() {
    local name=$1 bs=$2 nw=$3 vb=$4 mp=$5 stage=$6
    echo "---- $name: batch $bs, workers $nw, $vb, mixed precision $mp, stage $stage" >&2
    OUTPUT_DIR="$BENCH/runs/$name" BATCH_SIZE=$bs NUM_WORKERS=$nw VIDEO_BACKEND=$vb MIXED_PRECISION=$mp \
        STAGE_DATASET=$stage USAGE_CSV="$BENCH/$name.usage.csv" \
        bash "$REPO_ROOT/sim/train_act_hpc.sh" "$BENCH_STEPS" --save_checkpoint=false > "$BENCH/$name.log" 2>&1
    if [ $? -ne 0 ]; then echo "     FAILED, see $BENCH/$name.log" >&2; tail -5 "$BENCH/$name.log" >&2; echo 0; return; fi
    bash "$REPO_ROOT/sim/hpc_env.sh" python sim/tools/summarize_train_log.py --log "$BENCH/$name.log" \
        --batch-size "$bs" --log-freq "$LOG_FREQ" --usage "$BENCH/$name.usage.csv" --name "$name" \
        --full-steps $(( BASE_STEPS * BASE_BATCH / bs )) --json "$BENCH/$name.json" >&2 || { echo 0; return; }
    python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["wall_samples_per_s"] or 0)' "$BENCH/$name.json"
}
faster() { python3 -c 'import sys;print(1 if float(sys.argv[1])>float(sys.argv[2]) else 0)' "$1" "$2"; }

r1=$(run b8_w4_torchcodec_fp32 8 4 torchcodec no 0)
r2=$(run b8_w${MAXW}_torchcodec_fp32 8 $MAXW torchcodec no 0)
r3=$(run b8_w${MAXW}_pyav_fp32 8 $MAXW pyav no 0)
if [ "$(faster "$r3" "$r2")" = 1 ]; then VB=pyav; else VB=torchcodec; fi
echo "keep video backend: $VB" >&2

BEST_BS=8; BEST=$(python3 -c "print(max($r2,$r3))")
for bs in 32 64 128; do
    r=$(run b${bs}_w${MAXW}_${VB}_fp32 $bs $MAXW $VB no 0)
    if [ "$(faster "$r" "$BEST")" = 1 ]; then BEST=$r; BEST_BS=$bs; fi
done
echo "keep batch size: $BEST_BS" >&2

MP=no
r7=$(run b${BEST_BS}_w${MAXW}_${VB}_bf16 $BEST_BS $MAXW $VB bf16 0)
[ "$(faster "$r7" "$BEST")" = 1 ] && { MP=bf16; BEST=$r7; }
echo "keep mixed precision: $MP" >&2
run b${BEST_BS}_w${MAXW}_${VB}_${MP/no/fp32}_shm $BEST_BS $MAXW $VB $MP 1 >/dev/null

# ---- summary table ------------------------------------------------------------------------------
bash "$REPO_ROOT/sim/hpc_env.sh" python - "$BENCH" <<'EOF' | tee "$BENCH/summary.md"
import json, sys
from pathlib import Path
bench = Path(sys.argv[1])
rows = {r["name"]: r for r in (json.load(open(p)) for p in bench.glob("*.json"))}
order = [p.stem for p in sorted(bench.glob("*.log"), key=lambda p: p.stat().st_mtime)]
print("PROVISIONAL (synthetic dataset). Full-run estimate = same epochs as batch 8 x 20000 steps.\n")
print("| config | samples/s | steps/s | data_s | updt_s | GPU util | CPU cores | GPU mem | final loss | full run (steps) | est. time |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for n in order:
    r = rows.get(n)
    if r is None:
        print(f"| {n} | FAILED | | | | | | | | | |"); continue
    print(f"| {n} | {r['wall_samples_per_s']} | {r['wall_steps_per_s']} | {r['data_s']:.3f} | {r['updt_s']:.3f} | "
          f"{r.get('gpu_util_pct','?')}% | {r.get('cpu_cores','?')} | {r.get('gpu_mem_gib_max','?')} GiB | "
          f"{r['final_loss']:.3f} | {r['full_steps']} | {r['est_hours_full_run']:.2f} h |")
for n in order:
    if n.endswith("_bf16") and n.replace("_bf16", "_fp32") in rows and n in rows:
        a, b = rows[n.replace("_bf16", "_fp32")]["loss_trace"], rows[n]["loss_trace"]
        print(f"\nloss trace fp32 vs bf16 ({n}):\n  fp32 {[round(x, 3) for x in a]}\n  bf16 {[round(x, 3) for x in b]}")
EOF
echo "bench results: $BENCH/summary.md"
