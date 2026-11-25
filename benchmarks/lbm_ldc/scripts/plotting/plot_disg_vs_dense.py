#!/usr/bin/env python3
"""Plot the DenseDisg / Dense MLUPS ratio of a sweep that ran both grid layouts.

Reads the sweep YAML of ``lbm-sweep-run.sh`` and, for every ``strong_scaling``
experiment whose ``plots`` list ``disg_vs_dense``, pairs every run of a Disg grid with
the same run of its Dense twin:

  disg_distributed / dense_distributed   one panel per decomposition
  disg / dense                           one "single-process" panel

For every configs x precisions series and transport x hp-halo it writes into
``<data_dir>/plots/<experiment>/``

  <series>-disg-vs-dense-<transport>-hp<off|on>.png   ratio vs GPU count, one curve per N
  <series>-disg-vs-dense.csv                          every pair with both MLUPS

A ratio above 1 means the DenseDisg layout is faster. Both runs share G, so the ratio is
the same in MLUPS and in MLUPS per GPU.

Example::

    python3 benchmarks/lbm_ldc/scripts/plotting/plot_disg_vs_dense.py \\
        --config benchmarks/lbm_ldc/scripts/lbm-sweep.yml
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
    lr.add_common_args(p)
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
        exps, json_dir, plots_dir = lr.experiments(args, ("disg_vs_dense",))
    except (sc.SweepConfigError, ValueError) as e:
        sys.stderr.write(f"error: {e}\n")
        return 1
    if not exps:
        return 0
    wrote = False
    for exp in exps:
        wrote |= plot_experiment(exp, json_dir, plots_dir / exp["name"], args.show)
    return 0 if wrote else 1


def plot_experiment(exp: dict, json_dir: Path, out_dir: Path, show: bool) -> bool:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("viridis")
    sizes = exp["sizes"]
    wrote = False
    for series, *_ in sc.series_prefixes(exp):
        s = plot_scaling.Series(series, json_dir, exp,
                                (json_dir, series, exp["reference_label"]))
        twins = pairs(s)
        if not s.reports or not twins:
            sys.stderr.write(f"warning: {exp['name']}: {series}: no Disg / Dense pairs in "
                             f"{json_dir}\n")
            continue
        rows = []
        for t in exp["transports"]:
            for hp in exp["hp_halo"]:
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
                lr.save(fig, out_dir / f"{series}-disg-vs-dense-{t}-hp{hp}.png", show)
                wrote = True
        if not rows:
            continue
        csv = out_dir / f"{series}-disg-vs-dense.csv"
        with csv.open("w") as f:
            f.write("grid,decomposition,transport,hp_halo,n,g,dense_mlups,disg_mlups,speedup\n")
            for grid, d, t, hp, n, g, dense_m, disg_m in rows:
                f.write(f"{grid},{d},{t},{hp},{n},{g},{dense_m:.3f},{disg_m:.3f},"
                        f"{disg_m / dense_m:.4f}\n")
        print(f"summary saved to {csv}")
    return wrote


if __name__ == "__main__":
    sys.exit(main())
