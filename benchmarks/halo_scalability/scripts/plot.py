#!/usr/bin/env python3
"""Plot dense-vs-disg halo throughput from one or more CSV files.

The benchmark binary writes rows of the form
``grid,cube_edge,components,n_gpus,pair_count,iters,wall_ms_total,
wall_ms_per_iter,bytes_per_iter,gb_per_s``. This script renders three views:

  1. ``GB/s vs components`` at fixed N (one curve per grid kind).
  2. ``ms/iter vs components`` at fixed N (log-y; the disg flat-line vs the
     linear dense ramp is the headline finding).
  3. ``GB/s vs N`` at fixed C (one curve per grid kind, several Cs).

Pure stdlib + matplotlib + numpy (no pandas).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: Path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "grid": r["grid"],
                    # Older CSVs (pre-lattice) lacked the semantic / lattice columns.
                    "semantic": r.get("semantic", "standard"),
                    "lattice": r.get("lattice", "-"),
                    "N": int(r["cube_edge"]),
                    "C": int(r["components"]),
                    "n_gpus": int(r["n_gpus"]),
                    "pairs": int(r["pair_count"]),
                    "iters": int(r["iters"]),
                    "ms_total": float(r["wall_ms_total"]),
                    "ms_per_iter": float(r["wall_ms_per_iter"]),
                    "bytes_per_iter": float(r["bytes_per_iter"]),
                    "gb_per_s": float(r["gb_per_s"]),
                }
            )
    return rows


def plot_vs_components(rows, out: Path, title_suffix: str = ""):
    # Standard-semantic only -- the lattice variants pin Components to the lattice's Q.
    rows = [r for r in rows if r["semantic"] == "standard"]
    if not rows:
        return
    fixed_n = sorted({r["N"] for r in rows})
    n_to_plot = fixed_n[-1] if fixed_n else None
    if n_to_plot is None:
        return
    sub = [r for r in rows if r["N"] == n_to_plot]
    by_grid = defaultdict(list)
    for r in sub:
        by_grid[r["grid"]].append(r)
    for g in by_grid:
        by_grid[g].sort(key=lambda r: r["C"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"dense": "#d62728", "disg": "#1f77b4"}
    markers = {"dense": "o", "disg": "s"}

    ax = axes[0]
    for g, rs in by_grid.items():
        ax.plot([r["C"] for r in rs], [r["gb_per_s"] for r in rs],
                marker=markers.get(g, "x"), color=colors.get(g, None),
                label=f"carbon::{'Dense' if g == 'dense' else 'DenseDisg'}")
    ax.set_xlabel("components per cell")
    ax.set_ylabel("effective halo bandwidth (GB/s)")
    ax.set_title(f"Halo throughput vs components @ N={n_to_plot}{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    for g, rs in by_grid.items():
        ax.plot([r["C"] for r in rs], [r["ms_per_iter"] for r in rs],
                marker=markers.get(g, "x"), color=colors.get(g, None),
                label=f"carbon::{'Dense' if g == 'dense' else 'DenseDisg'}")
    ax.set_xlabel("components per cell")
    ax.set_ylabel("halo wall time (ms / iteration)")
    ax.set_yscale("log")
    ax.set_title(f"Halo wall time vs components @ N={n_to_plot}{title_suffix}")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def plot_vs_size(rows, out: Path, title_suffix: str = ""):
    rows = [r for r in rows if r["semantic"] == "standard"]
    if not rows:
        return
    cs = sorted({r["C"] for r in rows})
    by_grid_c = defaultdict(list)
    for r in rows:
        by_grid_c[(r["grid"], r["C"])].append(r)
    for k in by_grid_c:
        by_grid_c[k].sort(key=lambda r: r["N"])

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.colormaps.get("viridis")
    for i, c in enumerate(cs):
        color = cmap(i / max(len(cs) - 1, 1))
        for grid, ls in (("dense", "--"), ("disg", "-")):
            rs = by_grid_c.get((grid, c), [])
            if not rs:
                continue
            ax.plot([r["N"] for r in rs], [r["gb_per_s"] for r in rs],
                    linestyle=ls, marker="o" if grid == "dense" else "s",
                    color=color, alpha=0.9,
                    label=f"{'Dense' if grid == 'dense' else 'DenseDisg'} C={c}")
    ax.set_xlabel("cube edge N")
    ax.set_ylabel("effective halo bandwidth (GB/s)")
    ax.set_title(f"Halo throughput vs N{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def plot_speedup(rows, out: Path, title_suffix: str = ""):
    rows = [r for r in rows if r["semantic"] == "standard"]
    if not rows:
        return
    cs = sorted({r["C"] for r in rows})
    ns = sorted({r["N"] for r in rows})
    pivot = {}
    for r in rows:
        pivot[(r["grid"], r["N"], r["C"])] = r["ms_per_iter"]

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.colormaps.get("plasma")
    for i, c in enumerate(cs):
        xs, ys = [], []
        for n in ns:
            d = pivot.get(("dense", n, c))
            g = pivot.get(("disg", n, c))
            if d is None or g is None or g <= 0:
                continue
            xs.append(n)
            ys.append(d / g)
        if xs:
            ax.plot(xs, ys, marker="o", color=cmap(i / max(len(cs) - 1, 1)),
                    label=f"C={c}")
    ax.axhline(1.0, linestyle=":", color="grey")
    ax.set_xlabel("cube edge N")
    ax.set_ylabel("speedup (Dense ms/iter ÷ DenseDisg ms/iter)")
    ax.set_title(f"DenseDisg halo speedup{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def plot_lattice(rows, out: Path, title_suffix: str = ""):
    """Lattice halo throughput vs N, with a comparison to the matching-Q standard halo."""
    lattice_rows = [r for r in rows if r["semantic"] == "lattice"]
    if not lattice_rows:
        return
    qs = sorted({r["C"] for r in lattice_rows})
    by_label = defaultdict(list)
    for r in lattice_rows:
        label = f"{r['grid']} {r['lattice']}"
        by_label[(label, r["C"])].append(r)
    # Add the standard-halo curves at the matching Components for context.
    for r in rows:
        if r["semantic"] == "standard" and r["C"] in qs:
            label = f"{r['grid']} standard Q={r['C']}"
            by_label[(label, r["C"])].append(r)
    for k in by_label:
        by_label[k].sort(key=lambda r: r["N"])

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.colormaps.get("tab20")
    color_idx = 0
    for (label, _q), rs in sorted(by_label.items()):
        if not rs:
            continue
        style = "-" if "lattice" in label else "--"
        marker = "s" if "disg" in label else "o"
        ax.plot([r["N"] for r in rs], [r["ms_per_iter"] for r in rs],
                linestyle=style, marker=marker, color=cmap(color_idx % 20),
                alpha=0.9, label=label)
        color_idx += 1
    ax.set_xlabel("cube edge N")
    ax.set_ylabel("halo wall time (ms / iteration)")
    ax.set_yscale("log")
    ax.set_title(f"Lattice halo wall time vs N{title_suffix}")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def plot_lattice_speedup(rows, out: Path, title_suffix: str = ""):
    """Speedup of disg+lattice (best ordering) vs disg+standard at the same Q.

    Headline plot: shows how much wall time the lattice halo saves vs the
    standard halo on the same grid + cardinality count, for each lattice
    flavour.
    """
    pivot = {}
    for r in rows:
        if r["grid"] != "disg":
            continue
        if r["semantic"] == "standard":
            pivot[("std", r["N"], r["C"])] = r["ms_per_iter"]
        elif r["semantic"] == "lattice":
            pivot[(r["lattice"], r["N"], r["C"])] = r["ms_per_iter"]
    lattices = sorted({k[0] for k in pivot if k[0] != "std"})
    ns_all = sorted({k[1] for k in pivot})

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.colormaps.get("plasma")
    for i, latt in enumerate(lattices):
        q = 19 if "d3q19" in latt else 27
        xs, ys = [], []
        for n in ns_all:
            std = pivot.get(("std", n, q))
            lat = pivot.get((latt, n, q))
            if std is None or lat is None or lat <= 0:
                continue
            xs.append(n)
            ys.append(std / lat)
        if xs:
            ax.plot(xs, ys, marker="o", color=cmap(i / max(len(lattices) - 1, 1)),
                    label=f"{latt}")
    ax.axhline(1.0, linestyle=":", color="grey")
    # Theoretical upper bounds.
    ax.axhline(19.0 / 5.0, linestyle=":", color="#888888", alpha=0.4,
               label="D3Q19 max (19/5 = 3.8x)")
    ax.axhline(27.0 / 9.0, linestyle=":", color="#444444", alpha=0.4,
               label="D3Q27 max (27/9 = 3.0x)")
    ax.set_xlabel("cube edge N")
    ax.set_ylabel("speedup (disg standard ms/iter / disg lattice ms/iter)")
    ax.set_title(f"DenseDisg lattice halo speedup vs standard halo{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path,
                   help="CSV produced by carbon_benchmark_halo_scalability")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--title-suffix", default="")
    args = p.parse_args(argv)

    rows = read_csv(args.csv)
    if not rows:
        print(f"no rows in {args.csv}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_vs_components(rows, args.out_dir / "halo_vs_components.png",
                       title_suffix=args.title_suffix)
    plot_vs_size(rows, args.out_dir / "halo_vs_size.png",
                 title_suffix=args.title_suffix)
    plot_speedup(rows, args.out_dir / "halo_speedup_disg_over_dense.png",
                 title_suffix=args.title_suffix)
    plot_lattice(rows, args.out_dir / "halo_lattice_vs_size.png",
                 title_suffix=args.title_suffix)
    plot_lattice_speedup(rows, args.out_dir / "halo_lattice_speedup.png",
                         title_suffix=args.title_suffix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
