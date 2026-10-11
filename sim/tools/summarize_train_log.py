"""Throughput summary of a short lerobot-train run, from its log (and optionally a monitor CSV).

    python sim/tools/summarize_train_log.py --log train.log --batch-size 64 --log-freq 25 \
        [--usage usage.csv] [--skip 2] [--full-steps 20000] [--name bs64] [--json out.json]

LeRobot logs, every log_freq steps, the interval averages `updt_s` (forward+backward+optimizer, synced
by loss.item()) and `data_s` (waiting for the next batch + preprocessing). The first --skip intervals
(warm-up: worker start, cudnn autotune) are dropped. Step time = data_s + updt_s; steps/s and samples/s
are also computed from the wall-clock timestamps of the log lines as a cross-check (includes the
DataLoader re-spawning its workers at every epoch boundary, which the per-step averages also see).
"""

import argparse
import csv
import json
import re
import statistics
from datetime import datetime

LINE = re.compile(r"^INFO (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) .*?\bstep:\S+ .*?\bloss:(\S+) .*?updt_s:(\S+) data_s:(\S+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--log-freq", type=int, required=True)
    ap.add_argument("--usage")
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--full-steps", type=int, default=20000)
    ap.add_argument("--name", default="")
    ap.add_argument("--json")
    args = ap.parse_args()

    rows = []
    for line in open(args.log, errors="replace"):
        m = LINE.search(line.strip())
        if m:
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            rows.append((t, float(m.group(2)), float(m.group(3)), float(m.group(4))))
    if len(rows) <= args.skip + 1:
        raise SystemExit(f"only {len(rows)} log intervals in {args.log}; need > skip+1={args.skip + 1}")

    steady = rows[args.skip :]
    updt = statistics.fmean(r[2] for r in steady)
    data = statistics.fmean(r[3] for r in steady)
    step_s = updt + data
    wall = (steady[-1][0] - steady[0][0]) / ((len(steady) - 1) * args.log_freq)
    out = {
        "name": args.name,
        "batch_size": args.batch_size,
        "intervals_used": len(steady),
        "updt_s": round(updt, 4),
        "data_s": round(data, 4),
        "steps_per_s": round(1 / step_s, 2),
        "samples_per_s": round(args.batch_size / step_s, 1),
        "wall_steps_per_s": round(1 / wall, 2) if wall > 0 else None,
        "wall_samples_per_s": round(args.batch_size / wall, 1) if wall > 0 else None,
        "final_loss": steady[-1][1],
        "loss_trace": [r[1] for r in rows],
        "est_hours_full_run": round(args.full_steps * (wall if wall > 0 else step_s) / 3600, 2),
        "full_steps": args.full_steps,
    }
    if args.usage:
        t0, t1 = steady[0][0], steady[-1][0]
        u = [r for r in csv.DictReader(open(args.usage)) if t0 <= float(r["time"]) <= t1]
        if u:
            out["gpu_util_pct"] = round(statistics.fmean(float(r["gpu_util"]) for r in u), 1)
            out["gpu_mem_gib_max"] = round(max(float(r["gpu_mem_gib"]) for r in u), 1)
            out["cpu_cores"] = round(statistics.fmean(float(r["cpu_cores"]) for r in u), 1)
    print(json.dumps(out))
    if args.json:
        json.dump(out, open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
