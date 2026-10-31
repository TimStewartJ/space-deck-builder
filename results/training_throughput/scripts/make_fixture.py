"""Build a resumable benchmark fixture from an archived reference-recipe run.

Copies one checkpoint and every snapshot its self-play pool manifest names
into the fixture folder, rewrites the manifest paths to those copies, and
writes ``start.pth``. Training resumed from it plays the same opponent pool
the archived run had at that update.
"""
import argparse, shutil, sys
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--archive-models", required=True, nargs="+", help="folders holding the archived run's per-update checkpoints")
parser.add_argument("--out", required=True)
args = parser.parse_args()

out = Path(args.out)
(out / "pool").mkdir(parents=True, exist_ok=True)
ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
manifest = ckpt["pool_manifest"]
for entry in manifest["snapshots"]:
    candidates = [Path(folder) / Path(entry["path"]).name for folder in args.archive_models]
    source = next((c for c in candidates if c.exists()), candidates[0])
    if not source.exists():
        sys.exit(f"missing snapshot {source}")
    target = out / "pool" / source.name
    shutil.copy2(source, target)
    entry["path"] = str(target)
ckpt["pool_manifest"] = manifest
torch.save(ckpt, out / "start.pth")
print("update", ckpt.get("update"), "snapshots", [e["name"] for e in manifest["snapshots"]])
