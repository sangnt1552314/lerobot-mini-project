"""Check that MuJoCo can render offscreen here, and how fast (run on a GPU node inside the container).

    python sim_mujoco/tools/render_check.py --model <so101 scene.xml> [--frames 300] [--out frame.png]

Tries MUJOCO_GL=egl, then osmesa, each in a fresh subprocess, renders 640x480 frames and prints fps.
"""

import argparse
import json
import os
import subprocess
import sys
import time

CHILD = r"""
import json, os, sys, time
import numpy as np
import mujoco
model = mujoco.MjModel.from_xml_path(sys.argv[1])
data = mujoco.MjData(model)
r = mujoco.Renderer(model, height=480, width=640)
cams = [None] + [model.camera(i).name for i in range(model.ncam)]
n = int(sys.argv[2])
mujoco.mj_forward(model, data)
r.update_scene(data); img = r.render()
t = time.perf_counter()
for i in range(n):
    data.ctrl[:] = 0.3 * np.sin(i / 20)
    mujoco.mj_step(model, data)
    r.update_scene(data, camera=cams[i % len(cams)] or -1)
    img = r.render()
dt = time.perf_counter() - t
from PIL import Image
Image.fromarray(img).save(sys.argv[3])
vendor = renderer = "?"
try:
    from OpenGL import GL
    vendor = GL.glGetString(GL.GL_VENDOR).decode(); renderer = GL.glGetString(GL.GL_RENDERER).decode()
except Exception as e:
    renderer = f"(glGetString failed: {e})"
print(json.dumps({"fps": round(n / dt, 1), "shape": list(img.shape), "mean": float(img.mean()),
                  "gl_vendor": vendor, "gl_renderer": renderer}))
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--out", default="render_check.png")
    args = ap.parse_args()

    ok = False
    for backend in ("egl", "osmesa"):
        env = {**os.environ, "MUJOCO_GL": backend, "PYOPENGL_PLATFORM": backend}
        out = args.out.replace(".png", f"_{backend}.png")
        t = time.time()
        p = subprocess.run([sys.executable, "-c", CHILD, args.model, str(args.frames), out],
                           env=env, capture_output=True, text=True, timeout=600)
        if p.returncode == 0:
            res = json.loads(p.stdout.strip().splitlines()[-1])
            print(f"[ok]   MUJOCO_GL={backend}: {res}  (saved {out}, {time.time() - t:.1f}s incl. startup)")
            ok = True
        else:
            err = (p.stderr.strip().splitlines() or ["?"])[-3:]
            print(f"[FAIL] MUJOCO_GL={backend}: {' | '.join(err)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
