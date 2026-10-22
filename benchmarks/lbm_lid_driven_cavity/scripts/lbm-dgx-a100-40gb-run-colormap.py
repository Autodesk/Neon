#!/usr/bin/env python3
"""Discrete "best-family" colormap over (domain size x GPU count).

Companion plotter for ``lbm-dgx-a100-40gb-run-colormap.sh``.  For every
(domain size N, GPU count g) cell it looks at the four LBM families

    dense-intra  (Dense     grid, Peeraccess)   -> red
    dense-mpi    (Dense     grid, NCCL)          -> blue
    disg-intra   (DenseDisg grid, Peeraccess)    -> green
    disg-mpi     (DenseDisg grid, NCCL)          -> purple

and colours the cell with the family that achieves the best MLUPS, mixed
with white according to how many of the *other* families come within
``THRESHOLD`` (default 10%) of that best:

    k        = number of families (4)
    m        = number of OTHER families whose MLUPS >= best * (1 - THRESHOLD)
    white    = m / (k - 1)          # fraction of white blended in
    colour   = best_colour * (1 - white) + white_rgb * white

So:
  * m == 0  (every other family is >10% slower)  -> full family colour
  * m == 1                                        -> 1/3 white
  * m == 2                                        -> 2/3 white
  * m == k-1 (all others within 10%, i.e. a tie) -> full white

Reports use the shared ``lbm-dgx-a100-40gb-<label>-n<N>-g<G>.json`` naming
(produced by the run script) and are read from
``data/lbm-dgx-a100-40gb/json/``.  Cells that are missing any of the four
family reports are drawn as neutral grey.

Example::

    python3 benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-a100-40gb-run-colormap.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA = "carbon-lbm-report-v1"
PREFIX = "lbm-dgx-a100-40gb"

HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parent / "data" / PREFIX
DEFAULT_INPUT = ROOT_DIR / "json"
DEFAULT_OUTPUT = ROOT_DIR / f"{PREFIX}-run-colormap.png"

DEFAULT_SIZES = list(range(128, 609, 32))
DEFAULT_GPUS = [1, 2, 4, 8]

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
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    p.add_argument("--gpus", type=int, nargs="+", default=DEFAULT_GPUS)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Relative gap below which an 'other' family counts as a near-tie.",
    )
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def load_mlups(report_dir: Path, label: str, n: int, gpus: int) -> float | None:
    path = report_dir / f"{PREFIX}-{label}-n{n}-g{gpus}.json"
    if not path.is_file():
        return None
    with path.open() as f:
        raw = json.load(f)
    if raw.get("schema") != SCHEMA:
        sys.stderr.write(f"warning: skipping {path}: bad schema\n")
        return None
    return float(raw["results"]["mlups"])


def cell_color(
    values: dict[str, float],
    threshold: float,
) -> tuple[tuple[float, float, float], str, int]:
    """Return (rgb, winner_label, n_near_ties) for one cell."""
    winner = max(values, key=values.__getitem__)
    best = values[winner]
    palette = {label: rgb for (label, _name, rgb) in FAMILIES}
    base = palette.get(winner, (0.0, 0.0, 0.0))

    k = len(FAMILIES)
    near = 0
    for label, val in values.items():
        if label == winner:
            continue
        if val >= best * (1.0 - threshold):
            near += 1

    white_frac = near / (k - 1) if k > 1 else 0.0
    rgb = tuple(base[c] * (1.0 - white_frac) + white_frac for c in range(3))
    return rgb, winner, near


def main() -> int:
    args = parse_args()
    report_dir: Path = args.input_dir
    if not report_dir.is_dir():
        sys.stderr.write(f"error: input dir not found: {report_dir}\n")
        return 1

    sizes = args.sizes
    gpus = sorted(args.gpus)
    labels = [f[0] for f in FAMILIES]

    import numpy as np

    ny, nx = len(gpus), len(sizes)
    rgb_grid = np.ones((ny, nx, 3), dtype=float)
    winner_grid: list[list[str | None]] = [[None] * nx for _ in range(ny)]
    tie_grid = [[0] * nx for _ in range(ny)]
    missing = 0

    for iy, g in enumerate(gpus):
        for ix, n in enumerate(sizes):
            values: dict[str, float] = {}
            for label in labels:
                v = load_mlups(report_dir, label, n, g)
                if v is not None:
                    values[label] = v
            if len(values) < len(labels):
                rgb_grid[iy, ix] = MISSING_RGB
                missing += 1
                continue
            rgb, winner, near = cell_color(values, args.threshold)
            rgb_grid[iy, ix] = rgb
            winner_grid[iy][ix] = winner
            tie_grid[iy][ix] = near

    if missing:
        sys.stderr.write(f"warning: {missing} cell(s) missing one or more family reports (drawn grey)\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8.0, 0.7 * nx), max(3.0, 0.9 * ny) + 1.5))
    ax.imshow(rgb_grid, origin="lower", aspect="auto", interpolation="nearest")

    ax.set_xticks(range(nx))
    ax.set_xticklabels(
        [f"{n}\u00b3" for n in sizes],
        rotation=45 if nx > 7 else 0,
        ha="right" if nx > 7 else "center",
        fontsize=8 if nx > 7 else 10,
    )
    ax.set_yticks(range(ny))
    ax.set_yticklabels([f"{g} GPU" if g == 1 else f"{g} GPUs" for g in gpus])
    ax.set_xlabel("Domain Size")
    ax.set_ylabel("GPUs")

    # cell separators
    ax.set_xticks(np.arange(-0.5, nx, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, ny, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)

    # annotate each cell with the winning family + near-tie count
    for iy in range(ny):
        for ix in range(nx):
            w = winner_grid[iy][ix]
            if w is None:
                continue
            r, g_, b = rgb_grid[iy, ix]
            lum = 0.299 * r + 0.587 * g_ + 0.114 * b
            txt_color = "black" if lum > 0.55 else "white"
            near = tie_grid[iy][ix]
            ax.text(
                ix,
                iy,
                f"{FAMILY_ABBR[w]}\n({near})",
                ha="center",
                va="center",
                fontsize=6.5,
                color=txt_color,
                linespacing=0.9,
            )

    # legend: full family colours + white-mixing meaning
    handles = [mpatches.Patch(color=rgb, label=name) for (_, name, rgb) in FAMILIES]
    handles.append(mpatches.Patch(facecolor="white", edgecolor="0.6", label="white = near-tie (all within 10%)"))
    handles.append(mpatches.Patch(facecolor=MISSING_RGB, edgecolor="0.6", label="grey = data missing"))
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=8,
        frameon=False,
    )

    thr_pct = int(round(args.threshold * 100))
    fig.suptitle(
        "Carbon LBM best-family map \u2014 8\u00d7 A100 SXM4 (NVLink)\n"
        f"D3Q19+BGK, overlap=standard; cell = fastest family, whitened by #others within {thr_pct}% "
        "(abbrev + near-tie count)",
        fontsize=10,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))

    out: Path = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"plot saved to {out}")
    if args.show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
