"""Summarize a py-spy raw (folded-stack) profile.

Raw format is "stack;frames;…  <count>" per line. We walk each stack and
accumulate two numbers per frame:
  * self:  samples where the frame was the innermost (leaf)
  * total: samples where the frame appears anywhere on the stack (cumulative)

Usage: python scripts/analyze_pyspy.py <raw.txt> [--top 30]
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict


def analyze(path: str, top: int) -> None:
    self_counts: dict[str, int] = defaultdict(int)
    total_counts: dict[str, int] = defaultdict(int)
    total_samples = 0

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            m = re.match(r"^(.*) (\d+)$", line)
            if not m:
                continue
            stack, count_s = m.group(1), m.group(2)
            count = int(count_s)
            total_samples += count
            frames = stack.split(";")
            if frames:
                # Last frame is the leaf.
                self_counts[frames[-1]] += count
                for frame in set(frames):
                    total_counts[frame] += count

    print(f"Total samples: {total_samples}\n")
    print(f"--- Top {top} by SELF time (leaf) ---")
    for fn, c in sorted(self_counts.items(), key=lambda x: -x[1])[:top]:
        print(f"  {c:6d}  {c / total_samples:6.1%}  {fn}")
    print(f"\n--- Top {top} by TOTAL time (cumulative, excluding builtin idle/wait) ---")
    # Filter obvious idle frames for total view
    idle_hints = ("<idle>", "_wait", "poll", "acquire", "Empty:", "Semaphore")
    filtered = [
        (fn, c) for fn, c in total_counts.items()
        if not any(h in fn for h in idle_hints)
    ]
    for fn, c in sorted(filtered, key=lambda x: -x[1])[:top]:
        print(f"  {c:6d}  {c / total_samples:6.1%}  {fn}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    analyze(args.path, args.top)
