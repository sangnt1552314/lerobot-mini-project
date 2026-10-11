"""Sample GPU and CPU usage of a running training process into a CSV, until it exits.

    python sim/tools/monitor_usage.py --pid <lerobot-train pid> --out usage.csv

Columns: time (unix s), gpu_util (%), gpu_mem_gib, cpu_cores (CPU time of the process and all its
children, incl. DataLoader workers, per wall second), rss_gib. Uses NVML (same counters as
`nvidia-smi dmon`) for the GPU this job was given.
"""

import argparse
import os
import time

import psutil


def pick_gpu(pynvml):
    """The GPU named by CUDA_VISIBLE_DEVICES (index or UUID), else device 0."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if vis.startswith("GPU-") or vis.startswith("MIG-"):
        return pynvml.nvmlDeviceGetHandleByUUID(vis)
    return pynvml.nvmlDeviceGetHandleByIndex(int(vis) if vis.isdigit() else 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    import pynvml

    pynvml.nvmlInit()
    gpu = pick_gpu(pynvml)
    root = psutil.Process(args.pid)
    cpu_prev: dict[int, float] = {}
    t_prev = time.time()
    with open(args.out, "w") as f:
        f.write("time,gpu_util,gpu_mem_gib,cpu_cores,rss_gib\n")
        while root.is_running() and root.status() != psutil.STATUS_ZOMBIE:
            time.sleep(args.interval)
            now = time.time()
            procs = [root]
            try:
                procs += root.children(recursive=True)
            except psutil.NoSuchProcess:
                break
            cpu_now, rss = {}, 0
            for p in procs:
                try:
                    t = p.cpu_times()
                    cpu_now[p.pid] = t.user + t.system
                    rss += p.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            # CPU seconds used in this interval by processes alive at both samples (new workers count
            # from their start), divided by wall time = average busy cores.
            used = sum(c - cpu_prev.get(pid, 0.0) for pid, c in cpu_now.items())
            cores = used / (now - t_prev)
            cpu_prev, t_prev = cpu_now, now
            util = pynvml.nvmlDeviceGetUtilizationRates(gpu).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(gpu).used / 2**30
            f.write(f"{now:.1f},{util},{mem:.2f},{cores:.2f},{rss / 2**30:.2f}\n")
            f.flush()


if __name__ == "__main__":
    try:
        main()
    except psutil.NoSuchProcess:
        pass
