#!/usr/bin/env python3
"""Plot DenseDisgDistributed / DenseDistributed MLUPS speedup from a comparison sweep.

Expects reports named
  <prefix>-<grid>-<lattice>-<collision>-fp32-<decomp>-<transport>-hp<on|off>-n<N>-g<G>.json
as produced when GRIDS lists more than one grid (see lbm-dist-sweep-run.sh).

Usage:
  python3 plot_disg_vs_dense.py --json-dir .../data/lbm-disg-vs-dense/json --out-dir ...
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

NAME_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<grid>dense_distributed|disg_distributed)"
    r"-(?P<lattice>\w+)-(?P<collision>\w+)-fp32"
    r"-(?P<decomp>slab|pencil|brick)-(?P<transport>\w+)-hp(?P<hp>on|off)"
    r"-n(?P<n>\d+)-g(?P<g>\d+)\.json$"
)


def load_reports(json_dir: Path) -> dict:
    data = defaultdict(dict)  # (decomp, n, g) -> {grid: mlups}
    for path in sorted(json_dir.glob("*.json")):
        m = NAME_RE.match(path.name)
        if not m:
            continue
        with path.open() as f:
            rep = json.load(f)
        key = (m["decomp"], int(m["n"]), int(m["g"]))
        data[key][m["grid"]] = float(rep["results"]["mlups"])
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = load_reports(args.json_dir)
    if not data:
        raise SystemExit(f"no comparison reports in {args.json_dir}")

    decomps = sorted({k[0] for k in data})
    fig, axes = plt.subplots(1, len(decomps), figsize=(5 * len(decomps), 4), squeeze=False)
    for ax, decomp in zip(axes[0], decomps):
        ns = sorted({k[1] for k in data if k[0] == decomp})
        for n in ns:
            xs, ys = [], []
            for g in sorted({k[2] for k in data if k[0] == decomp and k[1] == n}):
                pair = data[(decomp, n, g)]
                if "dense_distributed" not in pair or "disg_distributed" not in pair:
                    continue
                xs.append(g)
                ys.append(pair["disg_distributed"] / pair["dense_distributed"])
            if xs:
                ax.plot(xs, ys, marker="o", label=f"N={n}")
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_title(decomp)
        ax.set_xlabel("GPUs")
        ax.set_ylabel("MLUPS(disg) / MLUPS(dense)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("DenseDisgDistributed vs DenseDistributed (hp-halo on)")
    fig.tight_layout()
    out = args.out_dir / "disg-vs-dense-speedup.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")

    # Also dump a CSV summary.
    csv = args.out_dir / "disg-vs-dense-summary.csv"
    with csv.open("w") as f:
        f.write("decomp,n,g,dense_mlups,disg_mlups,speedup\n")
        for (decomp, n, g), pair in sorted(data.items()):
            d = pair.get("dense_distributed")
            s = pair.get("disg_distributed")
            if d is None or s is None:
                continue
            f.write(f"{decomp},{n},{g},{d:.3f},{s:.3f},{s/d:.4f}\n")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
