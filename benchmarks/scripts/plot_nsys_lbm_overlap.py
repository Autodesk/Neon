#!/usr/bin/env python3
"""Render a zoomed CUDA timeline from an Nsight Systems SQLite export."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_events(sqlite_path: Path, t_min_s: float, t_max_s: float):
    con = sqlite3.connect(sqlite_path)
    t_min_ns = int(t_min_s * 1e9)
    t_max_ns = int(t_max_s * 1e9)

    kernels = con.execute(
        """
        SELECT k.start, k.end, k.deviceId, k.streamId, k.gridZ, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.demangledName
        WHERE k.start BETWEEN ? AND ?
          AND s.value LIKE '%make_iteration_kernel_d3q19%'
        ORDER BY k.start
        """,
        (t_min_ns, t_max_ns),
    ).fetchall()

    memcpys = con.execute(
        """
        SELECT start, end, deviceId, streamId, bytes, copyKind
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE start BETWEEN ? AND ?
        ORDER BY start
        """,
        (t_min_ns, t_max_ns),
    ).fetchall()
    con.close()
    return kernels, memcpys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--t-min", type=float, required=True)
    parser.add_argument("--t-max", type=float, required=True)
    parser.add_argument("--title", type=str, default="LBM D3Q19 OCC standard — 2 GPU zoom")
    args = parser.parse_args()

    kernels, memcpys = load_events(args.sqlite, args.t_min, args.t_max)
    if not kernels:
        raise SystemExit("no LBM kernels in window")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    y_map = {0: 2.0, 1: 1.0}
    lane_h = 0.35

    for start, end, dev, stream, grid_z, _name in kernels:
        t0 = start / 1e9
        dur_ms = (end - start) / 1e6
        label = "internal" if grid_z > 1 else "boundary"
        color = "#2a9d8f" if label == "internal" else "#e76f51"
        y = y_map[int(dev)] + (0.15 if label == "internal" else -0.15)
        ax.barh(y, dur_ms, left=t0 - args.t_min, height=lane_h, color=color, edgecolor="black",
                linewidth=0.4)
        ax.text(t0 - args.t_min + 0.02, y, f"{label} s{stream}", va="center", fontsize=7)

    for start, end, dev, stream, _bytes, copy_kind in memcpys:
        if copy_kind not in (1, 2, 8, 10):
            continue
        t0 = start / 1e9
        dur_ms = max((end - start) / 1e6, 0.004)
        y = y_map[int(dev)] - 0.55
        ax.barh(y, dur_ms, left=t0 - args.t_min, height=0.12, color="#457b9d", alpha=0.85,
                edgecolor="none")
        if dur_ms > 0.15:
            ax.text(t0 - args.t_min + 0.01, y, f"halo s{stream}", va="center", fontsize=6)

    ax.set_yticks([2.0, 1.0])
    ax.set_yticklabels(["GPU 0", "GPU 1"])
    ax.set_xlabel("Time (ms from window start)")
    ax.set_title(args.title)
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax.set_xlim(0, (args.t_max - args.t_min) * 1000)

    legend = [
        mpatches.Patch(color="#2a9d8f", label="LBM internal (DataView::Internal)"),
        mpatches.Patch(color="#e76f51", label="LBM boundary (DataView::Boundary)"),
        mpatches.Patch(color="#457b9d", label="Halo memcpy (auto_halo_pop_a stream)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
