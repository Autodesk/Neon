#!/usr/bin/env python3
"""Plot the DenseDisg / Dense MLUPS ratio of a sweep that ran both grid layouts.

Reads the sweep YAML of ``lbm-dgx-sweep-run.sh`` and its reports, and pairs every run
of a Disg grid with the same run of its Dense twin:

  disg_distributed / dense_distributed   one panel per decomposition
  disg / dense                           one "single-process" panel

For every configs x precisions series and transport x hp-halo it writes

  <series>-disg-vs-dense-<transport>-hp<off|on>.png   ratio vs GPU count, one curve per N
  <series>-disg-vs-dense.csv                          every pair with both MLUPS

A ratio above 1 means the DenseDisg layout is faster. Both runs share G, so the ratio is
the same in MLUPS and in MLUPS per GPU.

Example::

    python3 benchmarks/lbm_lid_driven_cavity/scripts/plotting/plot_disg_vs_dense.py \\
        --config benchmarks/lbm_lid_driven_cavity/data/lbm-dgx-sweep/lbm-dgx-sweep.yml
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
import plot_scaling  # noqa: E402

TWINS = {"disg": "dense", "disg_distributed": "dense_distributed"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot the DenseDisg / Dense MLUPS ratio.")
    p.add_argument("--config", type=Path, required=True, help="Sweep YAML of lbm-dgx-sweep-run.sh.")
    p.add_argument("--input-dir", type=Path, default=None,
                   help="Report directory (default: data/<prefix>/json).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="PNG / CSV directory (default: data/<prefix>/).")
    p.add_argument("--show", action="store_true", help="Pop each figure interactively.")
    return p.parse_args()


def pairs(s: plot_scaling.Series) -> list[tuple[plot_scaling.Panel, plot_scaling.Panel]]:
    """(disg panel, dense panel) of every decomposition both layouts ran."""
    panels = s.panels()
    return [((grid, d), (TWINS[grid], d)) for grid, d in panels
            if grid in TWINS and (TWINS[grid], d) in panels]


def pair_title(disg: plot_scaling.Panel) -> str:
    grid, decomp = disg
    if grid in sc.DISTRIBUTED_GRIDS:
        return f"Distributed, {lr.DECOMP_TITLES[decomp]}"
    return "Single-process"


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

    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("viridis")
    sizes = cfg["sizes"]
    wrote = False
    for series, *_ in sc.series_prefixes(cfg):
        s = plot_scaling.Series(series, report_dir, cfg, (report_dir, series, "dense-intra"))
        twins = pairs(s)
        if not s.reports or not twins:
            sys.stderr.write(f"warning: {series}: no Disg / Dense pairs in {report_dir}\n")
            continue
        rows = []
        for t in cfg["transports"]:
            for hp in cfg["hp_halo"]:
                ran = [(disg, dense) for disg, dense in twins
                       if (t, hp) in s.curves(disg) and (t, hp) in s.curves(dense)]
                if not ran:
                    continue
                fig, axes = plt.subplots(1, len(ran), figsize=(5 * len(ran), 4.3),
                                         sharey=True, squeeze=False)
                has_runs = False
                for ax, (disg, dense) in zip(axes[0], ran):
                    gpus = sorted(set(s.gpus(disg)) & set(s.gpus(dense)))
                    for i, n in enumerate(sizes):
                        xs, ys = [], []
                        for g in gpus:
                            a, b = s.get(disg, t, hp, n, g), s.get(dense, t, hp, n, g)
                            if a is None or b is None:
                                continue
                            xs.append(g)
                            ys.append(a.mlups / b.mlups)
                            rows.append((disg[0], disg[1], t, hp, n, g, b.mlups, a.mlups))
                        if xs:
                            has_runs = True
                            ax.plot(xs, ys, "o-", lw=1.5, ms=5, label=f"N={n}",
                                    color=cmap(0.85 * i / max(1, len(sizes) - 1)))
                    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
                    ax.set_title(pair_title(disg))
                    ax.set_xlabel("Number of GPUs")
                    if gpus:
                        ax.set_xticks(gpus)
                    ax.grid(True, ls=":", alpha=0.5)
                    ax.legend(loc="best", fontsize=8)
                if not has_runs:
                    plt.close(fig)
                    continue
                axes[0][0].set_ylabel("MLUPS(DenseDisg) / MLUPS(Dense)")
                fig.suptitle(f"DenseDisg vs Dense, {plot_scaling.curve_label(t, hp)}\n"
                             f"{s.describe()}", fontsize=10, y=1.02)
                fig.tight_layout()
                lr.save(fig, out_dir / f"{series}-disg-vs-dense-{t}-hp{hp}.png", args.show)
                wrote = True
        csv = out_dir / f"{series}-disg-vs-dense.csv"
        with csv.open("w") as f:
            f.write("grid,decomposition,transport,hp_halo,n,g,dense_mlups,disg_mlups,speedup\n")
            for grid, d, t, hp, n, g, dense_m, disg_m in rows:
                f.write(f"{grid},{d},{t},{hp},{n},{g},{dense_m:.3f},{disg_m:.3f},"
                        f"{disg_m / dense_m:.4f}\n")
        print(f"summary saved to {csv}")
    return 0 if wrote else 1


if __name__ == "__main__":
    sys.exit(main())
