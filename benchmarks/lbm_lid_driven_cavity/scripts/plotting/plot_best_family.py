#!/usr/bin/env python3
"""Discrete "best-family" colormap over (domain size x GPU count).

Reads the sweep YAML of ``lbm-dgx-sweep-run.sh`` and its reports. For every
(domain size N, GPU count g) cell of the sweep it looks at the four LBM families

    dense-intra  (Dense     grid, Peeraccess)   -> red
    dense-mpi    (Dense     grid, NCCL)          -> blue
    disg-intra   (DenseDisg grid, Peeraccess)    -> green
    disg-mpi     (DenseDisg grid, NCCL)          -> purple

and colours the cell with the family that achieves the best MLUPS, mixed
with white according to how many of the *other* families come within
``--threshold`` (default 10%) of that best:

    k        = number of families (4)
    m        = number of OTHER families whose MLUPS >= best * (1 - threshold)
    white    = m / (k - 1)          # fraction of white blended in
    colour   = best_colour * (1 - white) + white_rgb * white

So:
  * m == 0  (every other family is >10% slower)  -> full family colour
  * m == 1                                        -> 1/3 white
  * m == 2                                        -> 2/3 white
  * m == k-1 (all others within 10%, i.e. a tie) -> full white

Each cell is annotated with the winner, its MLUPS per GPU (in thousands, MLUPS / g)
and the near-tie count m. Cells missing any of the four reports are grey. The sweep
must run grids dense and disg with transports intra and mpi; set ``mpi_single_gpu``
to fill the 1-GPU row for the NCCL families.

For every configs x precisions series it writes ``<series>-best-family.png``.

Example::

    python3 benchmarks/lbm_lid_driven_cavity/scripts/plotting/plot_best_family.py \\
        --config benchmarks/lbm_lid_driven_cavity/data/lbm-dgx-a100-40gb/lbm-dgx-a100-40gb-colormap.yml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import lbm_reports as lr  # noqa: E402
import lbm_sweep_config as sc  # noqa: E402

# family label -> (display name, base RGB before white-mixing)
FAMILIES = [
    ("dense-intra", "Dense / Peeraccess", (0.839, 0.153, 0.157)),  # red   #d62728
    ("dense-mpi", "Dense / NCCL", (0.122, 0.467, 0.706)),          # blue  #1f77b4
    ("disg-intra", "DenseDisg / Peeraccess", (0.173, 0.627, 0.173)),  # green #2ca02c
    ("disg-mpi", "DenseDisg / NCCL", (0.580, 0.404, 0.741)),       # purple #9467bd
]
FAMILY_ABBR = {
    "dense-intra": "D\u00b7P",
    "dense-mpi": "D\u00b7N",
    "disg-intra": "Dg\u00b7P",
    "disg-mpi": "Dg\u00b7N",
}
MISSING_RGB = (0.85, 0.85, 0.85)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Best-family (size x GPU) colormap.")
    p.add_argument("--config", type=Path, required=True, help="Sweep YAML of lbm-dgx-sweep-run.sh.")
    p.add_argument("--input-dir", type=Path, default=None,
                   help="Report directory (default: data/<prefix>/json).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="PNG directory (default: data/<prefix>/).")
    p.add_argument("--threshold", type=float, default=0.10,
                   help="Relative gap below which an 'other' family counts as a near-tie.")
    p.add_argument("--show", action="store_true", help="Pop each figure interactively.")
    return p.parse_args()


def cell_color(
    values: dict[str, float],
    threshold: float,
) -> tuple[tuple[float, float, float], str, int]:
    """Return (rgb, winner_label, n_near_ties) for one cell."""
    winner = max(values, key=values.__getitem__)
    best = values[winner]
    palette = {label: rgb for (label, _name, rgb) in FAMILIES}
    base = palette[winner]

    k = len(FAMILIES)
    near = sum(1 for label, val in values.items()
               if label != winner and val >= best * (1.0 - threshold))
    white_frac = near / (k - 1)
    rgb = tuple(base[c] * (1.0 - white_frac) + white_frac for c in range(3))
    return rgb, winner, near


def plot_series(series: str, report_dir: Path, sizes: list[int], gpus: list[int],
                threshold: float, output: Path, show: bool) -> bool:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [f[0] for f in FAMILIES]
    ny, nx = len(gpus), len(sizes)
    rgb_grid = np.ones((ny, nx, 3), dtype=float)
    winner_grid: list[list[str | None]] = [[None] * nx for _ in range(ny)]
    tie_grid = [[0] * nx for _ in range(ny)]
    best_grid = [[0.0] * nx for _ in range(ny)]
    first: lr.Report | None = None
    missing = 0

    for iy, g in enumerate(gpus):
        for ix, n in enumerate(sizes):
            values: dict[str, float] = {}
            for label in labels:
                rep = lr.load(report_dir, series, label, n, g)
                if rep is not None:
                    values[label] = rep.mlups_per_gpu
                    first = first or rep
            if len(values) < len(labels):
                rgb_grid[iy, ix] = MISSING_RGB
                missing += 1
                continue
            rgb, winner, near = cell_color(values, threshold)
            rgb_grid[iy, ix] = rgb
            winner_grid[iy][ix] = winner
            tie_grid[iy][ix] = near
            best_grid[iy][ix] = values[winner]

    if first is None:
        sys.stderr.write(f"warning: no reports for {series} in {report_dir}\n")
        return False
    if missing:
        sys.stderr.write(f"warning: {series}: {missing} cell(s) missing one or more family "
                         "reports (drawn grey)\n")

    fig, ax = plt.subplots(figsize=(max(8.0, 0.7 * nx), max(3.0, 0.9 * ny) + 1.5))
    ax.imshow(rgb_grid, origin="lower", aspect="auto", interpolation="nearest")

    many = nx > 7
    ax.set_xticks(range(nx))
    ax.set_xticklabels([f"{n}\u00b3" for n in sizes], rotation=45 if many else 0,
                       ha="right" if many else "center", fontsize=8 if many else 10)
    ax.set_yticks(range(ny))
    ax.set_yticklabels([f"{g} GPU" if g == 1 else f"{g} GPUs" for g in gpus])
    ax.set_xlabel("Domain Size")
    ax.set_ylabel("GPUs")

    # cell separators
    ax.set_xticks(np.arange(-0.5, nx, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, ny, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)

    for iy in range(ny):
        for ix in range(nx):
            w = winner_grid[iy][ix]
            if w is None:
                continue
            r, g_, b = rgb_grid[iy, ix]
            lum = 0.299 * r + 0.587 * g_ + 0.114 * b
            ax.text(ix, iy,
                    f"{FAMILY_ABBR[w]}\n{best_grid[iy][ix] / 1000.0:.1f}k\n({tie_grid[iy][ix]})",
                    ha="center", va="center", fontsize=6.5,
                    color="black" if lum > 0.55 else "white", linespacing=0.9)

    thr_pct = int(round(threshold * 100))
    handles = [mpatches.Patch(color=rgb, label=name) for (_, name, rgb) in FAMILIES]
    handles.append(mpatches.Patch(facecolor="white", edgecolor="0.6",
                                  label=f"white = near-tie (all within {thr_pct}%)"))
    handles.append(mpatches.Patch(facecolor=MISSING_RGB, edgecolor="0.6",
                                  label="grey = data missing"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
              fontsize=8, frameon=False)

    fig.suptitle(
        f"Carbon LBM best-family map\n{lr.describe(first)}; cell = fastest family, whitened "
        f"by #others within {thr_pct}% (abbrev, its MLUPS per GPU in thousands, near-tie count)",
        fontsize=10, y=0.99)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    lr.save(fig, output, show)
    return True


def main() -> int:
    args = parse_args()
    try:
        cfg = sc.read_sweep_config(args.config)
        data_dir = lr.DATA_ROOT / cfg["prefix"]
        report_dir = lr.repo_dir(args.input_dir or data_dir / "json", must_exist=True)
        out_dir = lr.repo_dir(args.output_dir or data_dir, must_exist=False)
    except (sc.SweepConfigError, ValueError) as e:
        sys.stderr.write(f"error: {e}\n")
        return 1
    if not ({"dense", "disg"} <= set(cfg["grids"])
            and {"intra", "mpi"} <= set(cfg["transports"])):
        sys.stderr.write("error: the best-family map needs grids dense and disg with "
                         "transports intra and mpi\n")
        return 1

    wrote = False
    for series, *_ in sc.series_prefixes(cfg):
        wrote |= plot_series(series, report_dir, cfg["sizes"], sorted(cfg["gpus"]),
                             args.threshold, out_dir / f"{series}-best-family.png", args.show)
    return 0 if wrote else 1


if __name__ == "__main__":
    sys.exit(main())
