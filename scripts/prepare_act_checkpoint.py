"""Make a pretrained ACT checkpoint loadable as `lerobot-train --policy.path=<dir>` with the installed LeRobot.

    python scripts/prepare_act_checkpoint.py omiya239532/so101_act_cotrain --out /scratch/e1583535/checkpoints/act/so101_act_cotrain

omiya239532/so101_act_cotrain was saved by a newer LeRobot; 0.4.2 (the HPC training venv) rejects its
config.json ("The fields `use_peft`, `pretrained_revision` are not valid for ACTConfig"). This copies the
checkpoint (HF repo id or local dir) to --out with config.json reduced to the fields the installed
ACTConfig has (the dropped ones are printed and recorded in PREPARED_FROM.json), sets
pretrained_backbone_weights to null (the checkpoint already holds the trained backbone, so no ImageNet
download on an offline node), then loads the copy with strict=True plus its pre/post-processors. Weights
and normalization stats are copied unchanged. Re-running with the same source is a no-op (--force redoes it).
"""

import argparse
import dataclasses
import importlib.metadata as md
import json
import shutil
import sys
from pathlib import Path


def resolve_checkpoint(name):
    """Local pretrained_model / run dir, or a Hugging Face repo id (downloaded into HF_HOME)."""
    p = Path(name).expanduser()
    if p.exists():
        for cand in (p, p / "pretrained_model", p / "checkpoints" / "last" / "pretrained_model"):
            if (cand / "config.json").is_file():
                return cand.resolve()
        sys.exit(f"no config.json in {p}")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(name, allow_patterns=["config.json", "model.safetensors", "train_config.json",
                                                        "policy_*"]))


def sanitize_act_config(raw):
    """config.json dict -> (the fields the installed ACTConfig accepts, the dropped ones)."""
    from lerobot.policies.act.configuration_act import ACTConfig

    if raw.get("type", "act") != "act":
        sys.exit(f"not an ACT checkpoint (type={raw.get('type')!r})")
    fields = {f.name for f in dataclasses.fields(ACTConfig)}
    return ({k: v for k, v in raw.items() if k in fields},
            {k: v for k, v in raw.items() if k not in fields and k != "type"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="HF repo id or local checkpoint / run dir")
    ap.add_argument("--out", required=True, help="output pretrained_model dir")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    marker = out / "PREPARED_FROM.json"
    version = md.version("lerobot")
    src = resolve_checkpoint(args.source)
    print(f"source {args.source} -> {src}")
    stamp = fingerprint(src)
    if marker.is_file() and not args.force:
        prev = json.loads(marker.read_text())
        if prev.get("source") != args.source:
            sys.exit(f"{out} was prepared from {prev.get('source')}; use --force or another --out")
        if prev.get("lerobot") == version and prev.get("fingerprint") == stamp:
            print(f"{out} already prepared from {args.source} for LeRobot {version} (source unchanged)")
            verify(out)
            return
        print(f"{args.source} changed since {out} was prepared (or LeRobot did): preparing it again")
    raw = json.loads((src / "config.json").read_text())
    kept, dropped = sanitize_act_config(raw)
    kept = {"type": "act", **kept, "pretrained_backbone_weights": None}

    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if f.is_file() and f.name != "config.json":
            # train_config.json belongs to the source run (and LeRobot version): kept for reference only.
            name = "source_train_config.json" if f.name == "train_config.json" else f.name
            shutil.copy2(f, tmp / name)  # follows the HF cache symlinks
    (tmp / "config.json").write_text(json.dumps(kept, indent=4))
    (tmp / "PREPARED_FROM.json").write_text(json.dumps(
        {"source": args.source, "source_dir": str(src), "lerobot": version, "fingerprint": stamp,
         "dropped_config_fields": dropped,
         "changed": {"pretrained_backbone_weights": [raw.get("pretrained_backbone_weights"), None]}}, indent=2))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    print(f"wrote {out} (dropped config fields: {dropped or 'none'})")
    verify(out)


def fingerprint(src):
    """Identifies the source weights + config: resolved dir, sizes and modification times."""
    files = [src / "config.json", src / "model.safetensors"]
    return {"dir": str(src.resolve()),
            **{f.name: [f.stat().st_size, int(f.stat().st_mtime)] for f in files if f.exists()}}


def verify(path):
    """Load exactly as lerobot-train --policy.path does, plus strict weights and the processors."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(str(path))
    cfg.device = "cpu"
    policy = ACTPolicy.from_pretrained(str(path), config=cfg, strict=True)
    make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(path),
                             preprocessor_overrides={"device_processor": {"device": "cpu"}})
    n = sum(p.numel() for p in policy.parameters()) / 1e6
    feats = {k: list(v.shape) for k, v in policy.config.input_features.items()}
    print(f"verified: {n:.1f}M params, inputs {feats}, chunk {policy.config.chunk_size}, "
          f"lr {policy.config.optimizer_lr} (LeRobot {md.version('lerobot')})")


if __name__ == "__main__":
    main()
