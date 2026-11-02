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

    python3 benchmarks/lbm_lid_driven_cavity/scripts/plotting/lbm-n512-gpu-scaling-plot.py

With ``--config`` it reads the sweep YAML of ``lbm-dgx-sweep-run.sh`` instead and writes
both figures for every configs x precisions series and every size of that sweep::

    python3 benchmarks/lbm_lid_driven_cavity/scripts/plotting/lbm-n512-gpu-scaling-plot.py \\
        --config benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-sweep.yml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA = "carbon-lbm-report-v1"
PREFIX = "lbm-n512-scaling"
DEFAULT_N = 512
DEFAULT_GPUS = [1, 2, 4, 8]
DEFAULT_DIR = Path("benchmarks/lbm_lid_driven_cavity/data/lbm-n512-gpu-scaling")
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lbm_sweep_config  # noqa: E402

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
        "--config",
        type=Path,
        default=None,
        help="Sweep YAML of lbm-dgx-sweep-run.sh (e.g. scripts/lbm-dgx-sweep.yml): for every "
        "configs x precisions series and every size N, read data/<prefix>/json/ and write "
        "data/<prefix>/<series>-{eff,mlups}-vs-gpus-n<N>.png, with gpus from the file. "
        "--input-dir, --output-dir, --n and --gpus override it.",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=f"Report directory (default: {DEFAULT_DIR / 'json'}, or data/<prefix>/json with "
        "--config).",
    )
    p.add_argument("--output-eff", type=Path, default=None,
                   help=f"default: {DEFAULT_DIR / 'lbm-n512-gpu-scaling.png'}")
    p.add_argument("--output-mlups", type=Path, default=None,
                   help=f"default: {DEFAULT_DIR / 'lbm-n512-gpu-scaling-mlups.png'}")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="With --config: directory for the PNGs (default: data/<prefix>/).",
    )
    p.add_argument("--n", type=int, nargs="+", default=None,
                   help=f"Domain size(s) (default: {DEFAULT_N}, or sizes with --config).")
    p.add_argument("--gpus", type=int, nargs="+", default=None,
                   help="GPU counts (default: 1 2 4 8, or gpus with --config).")
    p.add_argument(
        "--prefix",
        default=None,
        help=f"Report filename prefix: <prefix>-<label>-n<N>-g<G>.json (default: {PREFIX}).",
    )
    p.add_argument("--show", action="store_true")
    args = p.parse_args()
    if args.config is not None and any(
        v is not None for v in (args.prefix, args.output_eff, args.output_mlups)
    ):
        p.error("--prefix and --output-* name a single series; with --config use --output-dir")
    if args.config is None:
        if args.output_dir is not None:
            p.error("--output-dir needs --config")
        if args.n is not None and len(args.n) > 1:
            p.error("several --n sizes need --config")
    return args


def load_mlups(report_dir: Path, prefix: str, label: str, n: int, gpus: int) -> float | None:
    path = report_dir / f"{prefix}-{label}-n{n}-g{gpus}.json"
    if not path.is_file():
        sys.stderr.write(f"warning: missing {path.name}\n")
        return None
    with path.open() as f:
        raw = json.load(f)
    if raw.get("schema") != SCHEMA:
        sys.stderr.write(f"warning: bad schema in {path.name}\n")
        return None
    return float(raw["results"]["mlups"])


def describe_config(path: Path) -> str:
    """Lattice, collision and precision of a report, e.g. ``D3Q27+KBC fp64``.

    Reports written before ``--precision`` existed have no ``precision`` key; they are fp32.
    """
    with path.open() as f:
        cfg = json.load(f).get("config", {})
    lattice = str(cfg.get("lattice", "q19")).upper()
    collision = str(cfg.get("collision", "bgk")).upper()
    precision = str(cfg.get("precision", "fp32"))
    return f"D3{lattice}+{collision} {precision}"


def collect_series(
    report_dir: Path,
    prefix: str,
    mapping: list[tuple[str, str]],
    n: int,
    gpus: list[int],
) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for key, label in mapping:
        pts: dict[int, float] = {}
        for g in gpus:
            mlups = load_mlups(report_dir, prefix, label, n, g)
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

    g_max = max(gpus)
    ideal_glups = baseline_g1 * g_max / 1000.0
    secax = ax.secondary_yaxis(
        "right",
        functions=(lambda e: e * ideal_glups, lambda g: g / ideal_glups),
    )
    if show_glups_ylabel:
        secax.set_ylabel(f"GLUPS (at {g_max} GPUs)")


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


def plot_scaling(
    report_dir: Path,
    prefix: str,
    n: int,
    gpus: list[int],
    output_eff: Path,
    output_mlups: Path,
    show: bool,
) -> int:
    """Render both figures for one series and size; returns 0 on success."""
    report_dir = report_dir.resolve()
    if not report_dir.is_dir():
        sys.stderr.write(f"error: input dir not found: {report_dir}\n")
        return 1

    baseline_g1 = load_mlups(report_dir, prefix, "dense-intra", n, 1)
    if baseline_g1 is None:
        sys.stderr.write(f"error: {prefix}: missing dense-intra g1 reference report for N={n}\n")
        return 1
    config_desc = describe_config(report_dir / f"{prefix}-dense-intra-n{n}-g1.json")

    dense = collect_series(report_dir, prefix, DENSE_SERIES, n, gpus)
    disg = collect_series(report_dir, prefix, DISG_SERIES, n, gpus)
    if not dense and not disg:
        sys.stderr.write("error: no reports loaded\n")
        return 1

    import matplotlib.pyplot as plt

    # --- Efficiency figure (Fig. 7 style) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    plot_efficiency_panel(
        axes[0], dense, gpus, baseline_g1, "Dense (SoA)", False
    )
    plot_efficiency_panel(
        axes[1], disg, gpus, baseline_g1, "DenseDisg", True
    )
    axes[0].set_ylabel("Parallel Efficiency")
    fig.suptitle(
        f"Carbon LBM strong scaling at N={n} \u2014 8\u00d7 A100 SXM4 (NVLink)\n"
        f"{config_desc}, overlap=standard; reference: dense-intra 1-GPU",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    output_eff.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_eff, dpi=150, bbox_inches="tight")
    print(f"plot saved to {output_eff.resolve()}")
    if show:
        plt.show()
    plt.close(fig)

    # --- MLUPS figure (same layout) ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    plot_mlups_panel(axes2[0], dense, gpus, baseline_g1, "Dense (SoA)")
    plot_mlups_panel(axes2[1], disg, gpus, baseline_g1, "DenseDisg")
    axes2[0].set_ylabel("MLUPS (10\u2076 lattice updates / s)")
    fig2.suptitle(
        f"Carbon LBM throughput vs GPU count at N={n} \u2014 8\u00d7 A100 SXM4 (NVLink)\n"
        f"{config_desc}, overlap=standard",
        fontsize=10,
        y=1.02,
    )
    fig2.tight_layout()
    fig2.savefig(output_mlups, dpi=150, bbox_inches="tight")
    print(f"plot saved to {output_mlups.resolve()}")
    if show:
        plt.show()
    plt.close(fig2)
    return 0


def main() -> int:
    args = parse_args()
    if args.config is None:
        return plot_scaling(
            args.input_dir or DEFAULT_DIR / "json",
            args.prefix or PREFIX,
            args.n[0] if args.n else DEFAULT_N,
            args.gpus or DEFAULT_GPUS,
            args.output_eff or DEFAULT_DIR / "lbm-n512-gpu-scaling.png",
            args.output_mlups or DEFAULT_DIR / "lbm-n512-gpu-scaling-mlups.png",
            args.show,
        )

    try:
        cfg = lbm_sweep_config.read_sweep_config(args.config)
    except lbm_sweep_config.SweepConfigError as e:
        sys.stderr.write(f"error: config: {e}\n")
        return 1
    data_dir = DATA_ROOT / cfg["prefix"]
    report_dir = args.input_dir or data_dir / "json"
    out_dir = args.output_dir or data_dir
    gpus = args.gpus or cfg["gpus"]
    rcs = [
        plot_scaling(
            report_dir,
            series,
            n,
            gpus,
            out_dir / f"{series}-eff-vs-gpus-n{n}.png",
            out_dir / f"{series}-mlups-vs-gpus-n{n}.png",
            args.show,
        )
        for series, *_ in lbm_sweep_config.series_prefixes(cfg)
        for n in (args.n or cfg["sizes"])
    ]
    return 0 if 0 in rcs else 1


if __name__ == "__main__":
    sys.exit(main())
