#!/usr/bin/env python3
"""Plot summary.csv from run_benchmarks.py."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    csv_path = args.input_dir / "summary.csv"
    if not csv_path.is_file():
        raise SystemExit(f"missing {csv_path}")

    rows: list[dict[str, str]] = []
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # block_sparse + skeleton_overlap MLUPS vs gpus @ n=256
    ax = axes[0]
    for bench in ("block_sparse", "skeleton_overlap"):
        sub = [
            r
            for r in rows
            if r["benchmark"] == bench
            and r["n"] == "256"
            and (bench != "skeleton_overlap" or r["overlap"] == "none")
        ]
        if not sub:
            continue
        xs = [int(r["gpus"]) for r in sub]
        ys = [float(r["metric_value"]) for r in sub]
        ax.plot(xs, ys, marker="o", label=bench)
    ax.set_xlabel("gpus")
    ax.set_ylabel("MLUPS")
    ax.set_title("MLUPS vs gpus (n=256, overlap=none for skeleton)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # skeleton_overlap all modes @ n=256
    ax = axes[1]
    sub = [r for r in rows if r["benchmark"] == "skeleton_overlap" and r["n"] == "256"]
    by_overlap: dict[str, list[tuple[int, float]]] = {}
    for r in sub:
        by_overlap.setdefault(r["overlap"], []).append((int(r["gpus"]), float(r["metric_value"])))
    for overlap, pts in sorted(by_overlap.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=overlap)
    ax.set_xlabel("gpus")
    ax.set_ylabel("MLUPS")
    ax.set_title("skeleton_overlap modes (n=256)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # reduce dense batch=32
    ax = axes[2]
    sub = [
        r
        for r in rows
        if r["benchmark"] == "reduce_throughput"
        and r["grid"] == "dense"
        and r["batch_iters"] == "32"
        and r["n"] == "256"
    ]
    if sub:
        xs = [int(r["gpus"]) for r in sub]
        ys = [float(r["metric_value"]) for r in sub]
        ax.plot(xs, ys, marker="o", label="dense batch=32")
    ax.set_xlabel("gpus")
    ax.set_ylabel("GB/s")
    ax.set_title("reduce_throughput dense (n=256, batch=32)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
