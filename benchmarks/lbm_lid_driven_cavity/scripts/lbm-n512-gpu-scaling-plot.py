#!/usr/bin/env python3
"""Plot N=512 strong scaling in Neon Fig. 7 style (Dense vs DenseDisg).

Companion to ``lbm-n512-gpu-scaling-run.sh``. Reads JSON reports from
``data/lbm-n512-gpu-scaling/json/`` and writes two figures:

  lbm-n512-gpu-scaling.png
      Fig. 7 style — two panels (Dense | DenseDisg), parallel efficiency vs
      GPU count with Peeraccess / NCCL / MPI (P2P); right panel carries a
      GLUPS secondary axis.

  lbm-n512-gpu-scaling-mlups.png
      Absolute throughput (MLUPS) vs GPU count, same two-panel layout.

Efficiency is normalised to a fixed serial baseline:

    eff(g) = MLUPS(config, N, g) / (MLUPS(dense-intra, N, 1) * g)

Example::

    python3 benchmarks/lbm_lid_driven_cavity/scripts/lbm-n512-gpu-scaling-plot.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA = "carbon-lbm-report-v1"
PREFIX = "lbm-n512-scaling"
DEFAULT_GPUS = [1, 2, 4, 8]

DENSE_SERIES = [
    ("Peeraccess", "dense-intra"),
    ("NCCL", "dense-mpi"),
    ("MPI (P2P)", "dense-mpip2p"),
]
DISG_SERIES = [
    ("Peeraccess", "disg-intra"),
    ("NCCL", "disg-mpi"),
    ("MPI (P2P)", "disg-mpip2p"),
]
SERIES_COLORS = {
    "Peeraccess": "#d62728",
    "NCCL": "#1f77b4",
    "MPI (P2P)": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot N=512 GPU strong scaling (Fig. 7 style).")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("benchmarks/lbm_lid_driven_cavity/data/lbm-n512-gpu-scaling/json"),
    )
    p.add_argument(
        "--output-eff",
        type=Path,
        default=Path(
            "benchmarks/lbm_lid_driven_cavity/data/lbm-n512-gpu-scaling/lbm-n512-gpu-scaling.png"
        ),
    )
    p.add_argument(
        "--output-mlups",
        type=Path,
        default=Path(
            "benchmarks/lbm_lid_driven_cavity/data/lbm-n512-gpu-scaling/lbm-n512-gpu-scaling-mlups.png"
        ),
    )
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--gpus", type=int, nargs="+", default=DEFAULT_GPUS)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def load_mlups(report_dir: Path, label: str, n: int, gpus: int) -> float | None:
    path = report_dir / f"{PREFIX}-{label}-n{n}-g{gpus}.json"
    if not path.is_file():
        sys.stderr.write(f"warning: missing {path.name}\n")
        return None
    with path.open() as f:
        raw = json.load(f)
    if raw.get("schema") != SCHEMA:
        sys.stderr.write(f"warning: bad schema in {path.name}\n")
        return None
    return float(raw["results"]["mlups"])


def collect_series(
    report_dir: Path,
    mapping: list[tuple[str, str]],
    n: int,
    gpus: list[int],
) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for key, label in mapping:
        pts: dict[int, float] = {}
        for g in gpus:
            mlups = load_mlups(report_dir, label, n, g)
            if mlups is not None:
                pts[g] = mlups
        if pts:
            out[key] = pts
    return out


def plot_efficiency_panel(
    ax,
    series: dict[str, dict[int, float]],
    gpus: list[int],
    baseline_g1: float,
    title: str,
    show_glups_ylabel: bool,
) -> None:
    for key, pts in series.items():
        xs = [g for g in gpus if g in pts]
        eff = [pts[g] / (baseline_g1 * g) for g in xs]
        ax.plot(
            xs,
            eff,
            "D-",
            color=SERIES_COLORS.get(key, None),
            lw=1.8,
            ms=6,
            label=key,
        )
    ax.set_title(title)
    ax.set_xlabel("Number of GPUs")
    ax.set_xticks(gpus)
    ax.set_ylim(0.0, 1.05)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)

    ideal_g8_glups = baseline_g1 * 8.0 / 1000.0
    secax = ax.secondary_yaxis(
        "right",
        functions=(lambda e: e * ideal_g8_glups, lambda g: g / ideal_g8_glups),
    )
    if show_glups_ylabel:
        secax.set_ylabel("GLUPS (at 8 GPUs)")


def plot_mlups_panel(
    ax,
    series: dict[str, dict[int, float]],
    gpus: list[int],
    baseline_g1: float,
    title: str,
) -> None:
    g_max = max(gpus)
    ax.plot(
        [1, g_max],
        [baseline_g1, baseline_g1 * g_max],
        "--",
        color="#888888",
        lw=1.0,
        alpha=0.7,
        label="Ideal linear",
    )
    for key, pts in series.items():
        xs = [g for g in gpus if g in pts]
        ys = [pts[g] for g in xs]
        ax.plot(
            xs,
            ys,
            "D-",
            color=SERIES_COLORS.get(key, None),
            lw=1.8,
            ms=6,
            label=key,
        )
    ax.set_title(title)
    ax.set_xlabel("Number of GPUs")
    ax.set_xticks(gpus)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)


def main() -> int:
    args = parse_args()
    report_dir = args.input_dir.resolve()
    if not report_dir.is_dir():
        sys.stderr.write(f"error: input dir not found: {report_dir}\n")
        return 1

    baseline_g1 = load_mlups(report_dir, "dense-intra", args.n, 1)
    if baseline_g1 is None:
        sys.stderr.write("error: missing dense-intra g1 reference report\n")
        return 1

    dense = collect_series(report_dir, DENSE_SERIES, args.n, args.gpus)
    disg = collect_series(report_dir, DISG_SERIES, args.n, args.gpus)
    if not dense and not disg:
        sys.stderr.write("error: no reports loaded\n")
        return 1

    import matplotlib.pyplot as plt

    # --- Efficiency figure (Fig. 7 style) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    plot_efficiency_panel(
        axes[0], dense, args.gpus, baseline_g1, "Dense (SoA)", False
    )
    plot_efficiency_panel(
        axes[1], disg, args.gpus, baseline_g1, "DenseDisg", True
    )
    axes[0].set_ylabel("Parallel Efficiency")
    fig.suptitle(
        f"Carbon LBM strong scaling at N={args.n} \u2014 8\u00d7 A100 SXM4 (NVLink)\n"
        "D3Q19+BGK, overlap=standard; reference: dense-intra 1-GPU",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    args.output_eff.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_eff, dpi=150, bbox_inches="tight")
    print(f"plot saved to {args.output_eff.resolve()}")
    if args.show:
        plt.show()
    plt.close(fig)

    # --- MLUPS figure (same layout) ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    plot_mlups_panel(axes2[0], dense, args.gpus, baseline_g1, "Dense (SoA)")
    plot_mlups_panel(axes2[1], disg, args.gpus, baseline_g1, "DenseDisg")
    axes2[0].set_ylabel("MLUPS (10\u2076 lattice updates / s)")
    fig2.suptitle(
        f"Carbon LBM throughput vs GPU count at N={args.n} \u2014 8\u00d7 A100 SXM4 (NVLink)\n"
        "D3Q19+BGK, overlap=standard",
        fontsize=10,
        y=1.02,
    )
    fig2.tight_layout()
    fig2.savefig(args.output_mlups, dpi=150, bbox_inches="tight")
    print(f"plot saved to {args.output_mlups.resolve()}")
    if args.show:
        plt.show()
    plt.close(fig2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
