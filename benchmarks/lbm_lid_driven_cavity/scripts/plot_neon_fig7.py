#!/usr/bin/env python3
"""Plot Neon Fig. 7 style parallel efficiency at 8 GPUs.

Compares Peeraccess (intra-process NVLink P2P) vs NCCL (MPI+NCCL) for
Dense and DenseDisg grid families.  Efficiency is normalized to a single
fixed serial baseline: the peak single-GPU dense throughput across the
sweep, times the GPU count (8):

    eff(N) = MLUPS(config, N, 8) / (max_N MLUPS(dense-intra, N, 1) * 8)

Because the baseline is a single constant, GLUPS is an exact linear
rescale of efficiency (GLUPS = eff * baseline * 8 / 1000), so one line
per series carries both and the right y-axis is an exact relabel of the
left.

Example::

    python3 benchmarks/lbm_lid_driven_cavity/scripts/plot_neon_fig7.py \\
        --input-dir benchmarks/lbm_lid_driven_cavity/data/lbm-dgx-a100-40gb/json \\
        --prefix lbm-dgx-a100-40gb \\
        --output benchmarks/lbm_lid_driven_cavity/data/lbm-dgx-a100-40gb/lbm-dgx-a100-40gb.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA = "carbon-lbm-report-v1"
DEFAULT_SIZES = list(range(128, 609, 32))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Neon Fig. 7 parallel efficiency.")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("reports/lbm_scaling/fig7"),
        help="Directory containing <prefix>-*.json reports.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <input-dir>/parallel_efficiency_8gpu.png).",
    )
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="Domain sizes N to include (default: 128..608 step 32).",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="fig7",
        help="Report filename prefix: <prefix>-<label>-n<N>-g<G>.json (default: fig7).",
    )
    p.add_argument("--show", action="store_true", help="Pop the figure interactively.")
    return p.parse_args()


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


def main() -> int:
    args = parse_args()
    report_dir: Path = args.input_dir
    if not report_dir.is_dir():
        sys.stderr.write(f"error: input dir not found: {report_dir}\n")
        return 1

    sizes = args.sizes
    prefix = args.prefix
    ref: dict[int, float] = {}
    for n in sizes:
        v = load_mlups(report_dir, prefix, "dense-intra", n, 1)
        if v is None:
            sys.stderr.write(f"warning: missing dense-intra g1 reference for N={n}\n")
        else:
            ref[n] = v

    if not ref:
        sys.stderr.write("error: no dense-intra g1 reference reports found\n")
        return 1

    ref_const = max(ref.values())
    ideal_8gpu = ref_const * 8.0

    eff_series: dict[str, dict[int, float]] = {
        "Peeraccess": {},
        "NCCL": {},
        "MPI (P2P)": {},
        "Peeraccess (Disg)": {},
        "NCCL (Disg)": {},
        "MPI (P2P) (Disg)": {},
    }
    mapping = [
        ("Peeraccess", "dense-intra", 8),
        ("NCCL", "dense-mpi", 8),
        ("MPI (P2P)", "dense-mpip2p", 8),
        ("Peeraccess (Disg)", "disg-intra", 8),
        ("NCCL (Disg)", "disg-mpi", 8),
        ("MPI (P2P) (Disg)", "disg-mpip2p", 8),
    ]
    for key, label, gpus in mapping:
        for n in sizes:
            if n not in ref:
                continue
            mlups = load_mlups(report_dir, prefix, label, n, gpus)
            if mlups is not None:
                eff_series[key][n] = mlups / ideal_8gpu

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

        ideal_8gpu_glups = ideal_8gpu / 1000.0
        secax = ax.secondary_yaxis(
            "right",
            functions=(lambda e: e * ideal_8gpu_glups, lambda g: g / ideal_8gpu_glups),
        )
        if show_glups_ylabel:
            secax.set_ylabel("GLUPS (8 GPUs)")

    plot_panel(axes[0], dense_peer_eff, dense_nccl_eff, dense_mpip2p_eff, "Dense (SoA)", False)
    plot_panel(axes[1], disg_peer_eff, disg_nccl_eff, disg_mpip2p_eff, "DenseDisg", True)
    axes[0].set_ylabel("Parallel Efficiency")

    fig.suptitle(
        "Carbon LBM parallel efficiency at 8 GPUs \u2014 8\u00d7 A100 SXM4 (NVLink)\n"
        "D3Q19+BGK, overlap=standard; reference: dense-intra 1-GPU",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()

    out = args.output or (report_dir / "parallel_efficiency_8gpu.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"plot saved to {out}")
    if args.show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
