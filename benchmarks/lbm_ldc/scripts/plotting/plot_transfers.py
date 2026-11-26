#!/usr/bin/env python3
"""Plot the rank x message transfer matrix written by ``lbm_ldc --export-transfers``.

The CSV (``rank,index,seq,kernel,group,op,peer,elements,bytes``) lists every cross-rank halo
message of one skeleton iteration, read from the compiled ``carbon::Skeleton``; the
``# key=value`` lines above its header (grid, decomposition, process grid, N, ...) make
the first line of the title. The plot
has one row per rank and one column per message in issue order; the colour is the
message size and each cell reads ``S>p`` (send to rank p) or ``R<p`` (receive from
rank p) with its size and element count. The right margin gives each rank's message count and bytes sent.

Example::

    mpiexec -n 8 carbon_benchmark_lbm_ldc --grid disg_distributed --n 512 \\
        --export-transfers transfers.csv --compile-only
    python3 benchmarks/lbm_ldc/scripts/plotting/plot_transfers.py transfers.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import lbm_reports as lr  # noqa: E402
import lbm_sweep_config as sc  # noqa: E402

COLUMNS = ("rank", "index", "seq", "kernel", "group", "op", "peer", "elements", "bytes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot the rank x message transfer matrix.")
    p.add_argument("csv", type=Path, help="CSV written by --export-transfers")
    p.add_argument("--out", type=Path, default=None,
                   help="output PNG (default: the CSV path with a .png suffix)")
    return p.parse_args()


def checked_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve and validate the command-line paths: an existing .csv in, a .png out."""
    src = args.csv.expanduser().resolve()
    if src.suffix.lower() != ".csv" or not src.is_file():
        raise ValueError(f"{args.csv}: not an existing .csv file")
    out = (args.out.expanduser().resolve() if args.out is not None
           else src.with_suffix(".png"))
    if out.suffix.lower() != ".png":
        raise ValueError(f"{out}: output must end in .png")
    if not out.parent.is_dir():
        raise ValueError(f"{out.parent}: output directory does not exist")
    return src, out


def read_messages(src: Path) -> tuple[dict[str, str], list[dict]]:
    """``# key=value`` lines above the header (the run description), then the messages."""
    meta: dict[str, str] = {}
    body: list[str] = []
    with src.open(newline="") as f:
        for line in f:
            if line.startswith("#"):
                key, sep, value = line[1:].strip().partition("=")
                if sep and key.replace("_", "").isalnum():
                    meta[key] = value
            else:
                body.append(line)
    reader = csv.DictReader(body)
    if tuple(reader.fieldnames or ()) != COLUMNS:
        raise ValueError(f"{src}: expected header {','.join(COLUMNS)}")
    rows = []
    for r in reader:
        if r["op"] not in ("send", "recv"):
            raise ValueError(f"{src}: bad op {r['op']!r}")
        rows.append({"rank": int(r["rank"]), "index": int(r["index"]),
                     "seq": int(r["seq"]), "kernel": r["kernel"],
                     "op": r["op"], "peer": int(r["peer"]), "elements": int(r["elements"]),
                     "bytes": int(r["bytes"])})
    return meta, rows


def run_title(meta: dict[str, str]) -> str:
    """One line describing the run, e.g. ``DenseDisgDistributed, slab (z) 1x1x8, N = 512``."""
    if "grid" not in meta:
        return ""
    parts = [lr.GRID_TITLES.get(meta["grid"], meta["grid"])]
    if meta["grid"] in sc.DISTRIBUTED_GRIDS and "decomposition" in meta:
        decomp = lr.DECOMP_TITLES.get(meta["decomposition"], meta["decomposition"])
        parts.append(f"{decomp} {meta.get('process_grid', '')}".strip())
    if "n" in meta:
        parts.append(f"N = {meta['n']}")
    if "ranks" in meta:
        parts.append(f"{meta['ranks']} ranks")
    model = " ".join(meta[k] for k in ("lattice", "collision", "precision") if k in meta)
    if model:
        parts.append(model)
    return ", ".join(parts)


def megabytes(n: int) -> str:
    return f"{n / 1e6:.2f} MB" if n >= 1_000_000 else f"{n / 1e3:.1f} KB"


def plot(rows: list[dict], title: str, out: Path) -> None:
    ranks = max(r["rank"] for r in rows) + 1
    cols = max(r["index"] for r in rows) + 1
    grid = [[math.nan] * cols for _ in range(ranks)]
    for r in rows:
        grid[r["rank"]][r["index"]] = r["bytes"] / 1e6

    width = max(7.0, 0.95 * cols + 4.5)
    fig, ax = plt.subplots(figsize=(width, max(3.0, 0.6 * ranks + 1.8)))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f2f2f2")
    vmax = max(r["bytes"] for r in rows) / 1e6
    im = ax.imshow(grid, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0.0, vmax=vmax)
    for r in rows:
        mb = r["bytes"] / 1e6
        arrow = f"S>{r['peer']}" if r["op"] == "send" else f"R<{r['peer']}"
        ax.text(r["index"], r["rank"], f"{arrow}\n{megabytes(r['bytes'])}\n{r['elements']:,} el", ha="center",
                va="center", fontsize=7, color="black" if mb > 0.6 * vmax else "white")

    totals = []
    for rank in range(ranks):
        mine = [r for r in rows if r["rank"] == rank]
        sent = sum(r["bytes"] for r in mine if r["op"] == "send")
        totals.append(f"{len(mine)} msgs, {megabytes(sent)} sent")

    ax.set_xticks(range(cols), [f"msg {i}" for i in range(cols)], fontsize=8)
    ax.set_yticks(range(ranks), [f"rank {i}" for i in range(ranks)])
    right = ax.secondary_yaxis("right")
    right.set_yticks(range(ranks), totals, fontsize=8)
    ax.set_xlabel("message (issue order)")
    ax.set_title(title, fontsize=10)
    # pad is a fraction of the axes width; keep ~1.7 in free for the right-hand totals.
    fig.colorbar(im, ax=ax, label="MB", pad=1.7 / (width - 2.8))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    try:
        src, out = checked_paths(args)
        meta, rows = read_messages(src)
    except (OSError, ValueError, KeyError) as e:
        sys.stderr.write(f"error: {e}\n")
        return 1
    if not rows:
        print(f"note: {src} has no cross-rank messages, nothing to plot")
        return 0
    kernels = sorted({(r["seq"], r["kernel"]) for r in rows})
    lines = [run_title(meta), "Cross-rank halo messages per skeleton iteration",
             ", ".join(f"#{seq} {name}" for seq, name in kernels)]
    title = "\n".join(line for line in lines if line)
    plot(rows, title, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
