#!/usr/bin/env python3
"""Plot Neon Fig. 7 style parallel efficiency at G GPUs (default 8).

Compares Peeraccess (intra-process NVLink P2P) vs NCCL (MPI+NCCL) for
Dense and DenseDisg grid families.  Efficiency is normalized to a single
fixed serial baseline: the peak single-GPU dense throughput across the
sweep, times the GPU count G:

    eff(N) = MLUPS(config, N, G) / (max_N MLUPS(dense-intra, N, 1) * G)

Because the baseline is a single constant, GLUPS is an exact linear
rescale of efficiency (GLUPS = eff * baseline * G / 1000), so one line
per series carries both and the right y-axis is an exact relabel of the
left.

Example::

    python3 benchmarks/lbm_lid_driven_cavity/scripts/plotting/plot_neon_fig7.py \\
        --input-dir benchmarks/lbm_lid_driven_cavity/data/lbm-dgx-a100-40gb/json \\
        --prefix lbm-dgx-a100-40gb \\
        --output benchmarks/lbm_lid_driven_cavity/data/lbm-dgx-a100-40gb/lbm-dgx-a100-40gb.png

With ``--config`` it reads the sweep YAML of ``lbm-dgx-sweep-run.sh`` instead and plots
every configs x precisions series of that sweep::

    python3 benchmarks/lbm_lid_driven_cavity/scripts/plotting/plot_neon_fig7.py \\
        --config benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-sweep.yml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA = "carbon-lbm-report-v1"
DEFAULT_SIZES = list(range(128, 609, 32))
DEFAULT_INPUT_DIR = Path("reports/lbm_scaling/fig7")
DEFAULT_PREFIX = "fig7"
DEFAULT_GPUS = 8
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lbm_sweep_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Neon Fig. 7 parallel efficiency.")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Sweep YAML of lbm-dgx-sweep-run.sh (e.g. scripts/lbm-dgx-sweep.yml): plot every "
        "configs x precisions series from data/<prefix>/json/ into "
        "data/<prefix>/<series>-eff-vs-n-g<G>.png, with sizes from the file and G = its "
        "largest gpus entry. --input-dir, --output-dir, --sizes and --gpus override it.",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=f"Directory containing <prefix>-*.json reports (default: {DEFAULT_INPUT_DIR}, "
        "or data/<prefix>/json with --config).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <input-dir>/parallel_efficiency_<G>gpu.png).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="With --config: directory for the PNGs (default: data/<prefix>/).",
    )
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=None,
        help="Domain sizes N to include (default: 128..608 step 32, or sizes with --config).",
    )
    p.add_argument(
        "--gpus",
        type=int,
        default=None,
        help=f"GPU count G of the efficiency curves (default: {DEFAULT_GPUS}, or the largest "
        "gpus entry with --config).",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default=None,
        help=f"Report filename prefix: <prefix>-<label>-n<N>-g<G>.json (default: {DEFAULT_PREFIX}).",
    )
    p.add_argument("--show", action="store_true", help="Pop the figure interactively.")
    args = p.parse_args()
    if args.config is not None and (args.prefix is not None or args.output is not None):
        p.error("--prefix and --output name a single series; with --config use --output-dir")
    if args.config is None and args.output_dir is not None:
        p.error("--output-dir needs --config")
    return args


def load_mlups(report_dir: Path, prefix: str, label: str, n: int, gpus: int) -> float | None:
    """Load MLUPS from ``<prefix>-<label>-n<N>-g<G>.json`` if present."""
    path = report_dir / f"{prefix}-{label}-n{n}-g{gpus}.json"
    if not path.is_file():
        return None
    with path.open() as f:
        raw = json.load(f)
    if raw.get("schema") != SCHEMA:
        sys.stderr.write(f"warning: skipping {path}: bad schema\n")
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


def plot_fig7(
    report_dir: Path,
    prefix: str,
    sizes: list[int],
    gpus: int,
    output: Path | None,
    show: bool,
) -> int:
    """Render one series (``<prefix>-<label>-n<N>-g<G>.json``); returns 0 on success."""
    if not report_dir.is_dir():
        sys.stderr.write(f"error: input dir not found: {report_dir}\n")
        return 1

    ref: dict[int, float] = {}
    for n in sizes:
        v = load_mlups(report_dir, prefix, "dense-intra", n, 1)
        if v is None:
            sys.stderr.write(f"warning: {prefix}: missing dense-intra g1 reference for N={n}\n")
        else:
            ref[n] = v

    if not ref:
        sys.stderr.write(f"error: {prefix}: no dense-intra g1 reference reports found\n")
        return 1

    ref_const = max(ref.values())
    ideal = ref_const * gpus
    config_desc = describe_config(report_dir / f"{prefix}-dense-intra-n{min(ref)}-g1.json")

    eff_series: dict[str, dict[int, float]] = {
        "Peeraccess": {},
        "NCCL": {},
        "MPI (P2P)": {},
        "Peeraccess (Disg)": {},
        "NCCL (Disg)": {},
        "MPI (P2P) (Disg)": {},
    }
    mapping = [
        ("Peeraccess", "dense-intra"),
        ("NCCL", "dense-mpi"),
        ("MPI (P2P)", "dense-mpip2p"),
        ("Peeraccess (Disg)", "disg-intra"),
        ("NCCL (Disg)", "disg-mpi"),
        ("MPI (P2P) (Disg)", "disg-mpip2p"),
    ]
    for key, label in mapping:
        for n in sizes:
            if n not in ref:
                continue
            mlups = load_mlups(report_dir, prefix, label, n, gpus)
            if mlups is not None:
                eff_series[key][n] = mlups / ideal

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    dense_peer_eff = [(n, eff_series["Peeraccess"][n]) for n in sizes if n in eff_series["Peeraccess"]]
    dense_nccl_eff = [(n, eff_series["NCCL"][n]) for n in sizes if n in eff_series["NCCL"]]
    dense_mpip2p_eff = [(n, eff_series["MPI (P2P)"][n]) for n in sizes if n in eff_series["MPI (P2P)"]]
    disg_peer_eff = [(n, eff_series["Peeraccess (Disg)"][n]) for n in sizes if n in eff_series["Peeraccess (Disg)"]]
    disg_nccl_eff = [(n, eff_series["NCCL (Disg)"][n]) for n in sizes if n in eff_series["NCCL (Disg)"]]
    disg_mpip2p_eff = [
        (n, eff_series["MPI (P2P) (Disg)"][n]) for n in sizes if n in eff_series["MPI (P2P) (Disg)"]
    ]

    def plot_panel(
        ax,
        peer_eff,
        nccl_eff,
        mpip2p_eff,
        title: str,
        show_glups_ylabel: bool,
    ) -> None:
        peer_color = "#d62728"
        nccl_color = "#1f77b4"
        mpip2p_color = "#2ca02c"
        if peer_eff:
            ax.plot(
                [p[0] for p in peer_eff],
                [p[1] for p in peer_eff],
                "D-",
                color=peer_color,
                lw=1.8,
                ms=6,
                label="Peeraccess",
            )
        if nccl_eff:
            ax.plot(
                [p[0] for p in nccl_eff],
                [p[1] for p in nccl_eff],
                "D-",
                color=nccl_color,
                lw=1.8,
                ms=6,
                label="NCCL",
            )
        if mpip2p_eff:
            ax.plot(
                [p[0] for p in mpip2p_eff],
                [p[1] for p in mpip2p_eff],
                "D-",
                color=mpip2p_color,
                lw=1.8,
                ms=6,
                label="MPI (P2P)",
            )
        ax.set_title(title)
        ax.set_xlabel("Domain Size")
        xticks = [p[0] for p in (peer_eff or nccl_eff or mpip2p_eff)]
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            [f"{n}\u00b3" for n in xticks],
            rotation=45 if len(xticks) > 7 else 0,
            ha="right" if len(xticks) > 7 else "center",
            fontsize=8 if len(xticks) > 7 else 10,
        )
        ax.set_ylim(0.0, 1.05)
        ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="lower right", fontsize=9)

        ideal_glups = ideal / 1000.0
        secax = ax.secondary_yaxis(
            "right",
            functions=(lambda e: e * ideal_glups, lambda g: g / ideal_glups),
        )
        if show_glups_ylabel:
            secax.set_ylabel(f"GLUPS ({gpus} GPUs)")

    plot_panel(axes[0], dense_peer_eff, dense_nccl_eff, dense_mpip2p_eff, "Dense (SoA)", False)
    plot_panel(axes[1], disg_peer_eff, disg_nccl_eff, disg_mpip2p_eff, "DenseDisg", True)
    axes[0].set_ylabel("Parallel Efficiency")

    fig.suptitle(
        f"Carbon LBM parallel efficiency at {gpus} GPUs \u2014 8\u00d7 A100 SXM4 (NVLink)\n"
        f"{config_desc}, overlap=standard; reference: dense-intra 1-GPU",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()

    out = output or (report_dir / f"parallel_efficiency_{gpus}gpu.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"plot saved to {out}")
    if show:
        plt.show()
    plt.close(fig)
    return 0


def main() -> int:
    args = parse_args()
    if args.config is None:
        return plot_fig7(
            args.input_dir or DEFAULT_INPUT_DIR,
            args.prefix or DEFAULT_PREFIX,
            args.sizes or DEFAULT_SIZES,
            args.gpus or DEFAULT_GPUS,
            args.output,
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
    sizes = args.sizes or cfg["sizes"]
    gpus = args.gpus or max(cfg["gpus"])
    rcs = [
        plot_fig7(report_dir, series, sizes, gpus, out_dir / f"{series}-eff-vs-n-g{gpus}.png",
                  args.show)
        for series, *_ in lbm_sweep_config.series_prefixes(cfg)
    ]
    return 0 if 0 in rcs else 1


if __name__ == "__main__":
    sys.exit(main())
