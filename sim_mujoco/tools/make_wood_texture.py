"""Seamless straight-grain wood texture for a setup's table (Appearance.table_texture).

    python sim_mujoco/tools/make_wood_texture.py --rgb 126 120 120 --out sim_mujoco/assets/office/table.png

--rgb is the mean colour the texture should have: measure the desk on the wrist photo (truest colours) and
scale it down for the sim's lighting (the desk renders ~1.5x brighter than the photo, so x0.67 is a good
start; check with sim_mujoco/tools/overlay_photo.py). --grain is the grain contrast (std / mean on the photo).
"""

import argparse

import numpy as np
from PIL import Image


def wood(rgb, grain_contrast=0.035, fine_contrast=0.012, size=1024, seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.arange(size)[:, None] / size
    x = np.arange(size)[None, :] / size
    yw = y + 0.004 * np.sin(2 * np.pi * 2 * x + 1.0) + 0.002 * np.sin(2 * np.pi * 5 * x + 2.0)   # wavy grain lines
    grain = np.zeros((size, size))
    for k in rng.integers(20, 260, 90):            # integer cycles per tile -> seamless
        grain += rng.normal() / np.sqrt(k / 20) * np.sin(2 * np.pi * k * yw + rng.uniform(0, 2 * np.pi))
    grain /= grain.std()
    fine = rng.normal(size=(size, size))
    freq = np.hypot(*np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size)))
    fine = np.real(np.fft.ifft2(np.fft.fft2(fine) * (freq < 0.15)))
    fine /= fine.std()
    lum = 1 + grain_contrast * grain + fine_contrast * fine
    return np.clip(np.asarray(rgb, float)[None, None, :] * lum[..., None], 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rgb", type=float, nargs=3, required=True, help="mean colour, 0..255")
    ap.add_argument("--grain", type=float, default=0.035, help="grain contrast")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    img = wood(args.rgb, args.grain, seed=args.seed)
    Image.fromarray(img).save(args.out, optimize=True)
    print(f"{args.out}: mean {img.reshape(-1, 3).mean(0).round(1)}, std {img.reshape(-1, 3).std(0).round(1)}")


if __name__ == "__main__":
    main()
