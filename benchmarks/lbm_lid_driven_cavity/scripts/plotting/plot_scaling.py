#!/usr/bin/env python3
"""Plot LBM strong scaling as MLUPS per GPU.

One panel per grid (``dense``, ``disg``) or distributed grid x decomposition
(``dense_distributed`` slab | pencil | brick, ...), one curve per transport x hp-halo.
The y axis is always ``results.mlups / G``, so perfect strong scaling is a horizontal
line: the dashed "ideal" line is the 1-GPU reference run of the same size N.

``--config`` is a sweep YAML of ``lbm-dgx-sweep-run.sh``. For every configs x precisions
series ``<series> = <prefix>-<lattice>-<collision>-<precision>`` it writes the figures
the YAML's ``plots`` key lists (default: all three):

  vs_gpus     <series>-mlups-per-gpu-vs-gpus-n<N>.png  MLUPS/GPU vs GPU count, per size N
  vs_n        <series>-mlups-per-gpu-vs-n-g<G>.png     MLUPS/GPU vs N at the largest GPU
                                                       count G, against the per-N reference
  hp_speedup  <series>-hp-speedup-<t>.png              MLUPS(hp-halo on) / MLUPS(hp-halo
                                                       off) vs GPU count, one curve per N
                                                       (when hp_halo has both)
  grids       <series>-grids-n<N>.png                  MLUPS/GPU vs GPU count, one curve
                                                       per grid: left the same transport
                                                       (NCCL, slab, hp-halo off), right each
                                                       grid's fastest run per GPU count,
                                                       tagged with its configuration (when
                                                       the sweep has two or more grids)

The reference of size N is ``<reference_dir>/<reference_prefix>-<lattice>-<collision>-
<precision>-<reference_label>-n<N>-g1.json``, by default the sweep's own 1-GPU
dense-intra run (see ``lbm_sweep_config.reference``). Distributed panels label each GPU
count with the process grid ProcessGrid chose (x*y*z).

Example::

    python3 benchmarks/lbm_lid_driven_cavity/scripts/plotting/plot_scaling.py \\
        --config benchmarks/lbm_lid_driven_cavity/data/lbm-dgx-sweep/lbm-dgx-sweep.yml
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import lbm_reports as lr  # noqa: E402
import lbm_sweep_config as sc  # noqa: E402

Panel = tuple[str, str]  # (grid, decomposition)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot LBM strong scaling as MLUPS per GPU.")
    p.add_argument("--config", type=Path, required=True,
                   help="Sweep YAML of lbm-dgx-sweep-run.sh: plot every series of that sweep "
                   "from data/<prefix>/json/ into data/<prefix>/.")
    p.add_argument("--input-dir", type=Path, default=None,
                   help="Report directory (default: data/<prefix>/json).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="PNG directory (default: data/<prefix>/).")
    p.add_argument("--sizes", type=int, nargs="+", default=None,
                   help="Domain sizes N (default: the config's sizes).")
    p.add_argument("--show", action="store_true", help="Pop each figure interactively.")
    return p.parse_args()


class Series:
    """Every report of one series, indexed by (label, n, g), plus its 1-GPU references."""

    def __init__(self, name: str, report_dir: Path, cfg: dict,
                 reference: tuple[Path, str, str]) -> None:
        self.name = name
        self.cfg = cfg
        self.cells = sc.runs(cfg)
        ref_dir, ref_series, ref_label = reference
        self.ref_name = f"{ref_series} {ref_label}"
        self.reports: dict[tuple[str, int, int], lr.Report] = {}
        self.refs: dict[int, lr.Report] = {}
        for n in cfg["sizes"]:
            ref = lr.load(ref_dir, ref_series, ref_label, n, 1)
            if ref is not None:
                self.refs[n] = ref
            for grid, decomp, t, hp, g in self.cells:
                label = sc.label(grid, decomp, t, hp)
                rep = lr.load(report_dir, name, label, n, g)
                if rep is not None:
                    self.reports[(label, n, g)] = rep

    def panels(self) -> list[Panel]:
        return list(dict.fromkeys((grid, decomp) for grid, decomp, *_ in self.cells))

    def curves(self, panel: Panel) -> list[tuple[str, str]]:
        return list(dict.fromkeys((t, hp) for grid, decomp, t, hp, _ in self.cells
                                  if (grid, decomp) == panel))

    def gpus(self, panel: Panel) -> list[int]:
        return sorted({g for grid, decomp, *_, g in self.cells if (grid, decomp) == panel})

    def max_gpus(self) -> int:
        return max(g for *_, g in self.cells)

    def get(self, panel: Panel, t: str, hp: str, n: int, g: int) -> lr.Report | None:
        return self.reports.get((sc.label(*panel, t, hp), n, g))

    def ref_mlups(self, n: int) -> float | None:
        ref = self.refs.get(n)
        return None if ref is None else ref.mlups

    def process_grid(self, panel: Panel, g: int) -> str | None:
        labels = {sc.label(*panel, t, hp) for t, hp in self.curves(panel)}
        for (label, _, gg), rep in self.reports.items():
            if gg == g and label in labels:
                return rep.process_grid
        return None

    def describe(self) -> str:
        return lr.describe(next(iter(self.reports.values())))


def panel_title(panel: Panel) -> str:
    grid, decomp = panel
    if grid in sc.DISTRIBUTED_GRIDS:
        return f"{lr.GRID_TITLES[grid]}, {lr.DECOMP_TITLES[decomp]}"
    return lr.GRID_TITLES[grid]


def curve_label(t: str, hp: str) -> str:
    return lr.TRANSPORT_NAMES[t] + (" + hp-halo" if hp == "on" else "")


def subplots(n_panels: int):
    import matplotlib.pyplot as plt

    ncols = n_panels if n_panels <= 3 else (2 if n_panels == 4 else 3)
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.3 * nrows), sharey=True,
                             squeeze=False)
    flat = list(axes.flat)
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel(lr.MLUPS_PER_GPU_LABEL)
    return fig, flat[:n_panels]


def plot_curve(ax, pts: dict[int, float], t: str, hp: str) -> None:
    xs = sorted(pts)
    marker, ls = lr.HP_STYLES[hp]
    ax.plot(xs, [pts[x] for x in xs], marker=marker, ls=ls, lw=1.8, ms=5,
            color=lr.TRANSPORT_COLORS[t], label=curve_label(t, hp))


def gpu_axis(ax, s: Series, panel: Panel, gpus: list[int]) -> None:
    ax.set_xticks(gpus)
    if panel[0] in sc.DISTRIBUTED_GRIDS:
        labels = []
        for g in gpus:
            pg = "ref" if g == 1 else s.process_grid(panel, g)
            labels.append(f"{g}\n{pg}" if pg else str(g))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("Number of GPUs (process grid x*y*z)")
    else:
        ax.set_xlabel("Number of GPUs")
    ax.set_xlim(0.5, s.max_gpus() + 0.5)


def size_axis(ax, sizes: list[int]) -> None:
    many = len(sizes) > 7
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{n}\u00b3" for n in sizes], rotation=45 if many else 0,
                       ha="right" if many else "center", fontsize=8 if many else 10)
    ax.set_xlabel("Domain size")


def plot_vs_gpus(s: Series, n: int, output: Path, show: bool) -> bool:
    ref = s.ref_mlups(n)
    if ref is None:
        sys.stderr.write(f"warning: {s.name}: no 1-GPU reference ({s.ref_name}) for N={n}, "
                         "skipping\n")
        return False
    panels = s.panels()
    fig, axes = subplots(len(panels))
    g_max = s.max_gpus()
    for ax, panel in zip(axes, panels):
        ax.plot([1, g_max], [ref, ref], "--", color="#888888", lw=1.0,
                label=f"Ideal (1 GPU: {ref:.0f})")
        gpus = s.gpus(panel)
        if 1 not in gpus:
            ax.plot([1], [ref], "*", color="black", ms=11, label="1-GPU reference")
        for t, hp in s.curves(panel):
            pts = {g: rep.mlups_per_gpu for g in gpus
                   if (rep := s.get(panel, t, hp, n, g)) is not None}
            if pts:
                plot_curve(ax, pts, t, hp)
        ax.set_title(panel_title(panel))
        ax.set_ylim(bottom=0.0, top=max(ax.get_ylim()[1], ref * 1.15))
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="lower left", fontsize=8)
        gpu_axis(ax, s, panel, sorted({1, *gpus}))
    fig.suptitle(f"Carbon LBM strong scaling at N={n}: MLUPS per GPU\n"
                 f"{s.describe()}; ideal: 1-GPU {s.ref_name}", fontsize=10, y=1.02)
    fig.tight_layout()
    lr.save(fig, output, show)
    return True


def plot_vs_n(s: Series, g: int, output: Path, show: bool) -> bool:
    sizes = [n for n in s.cfg["sizes"] if s.ref_mlups(n) is not None]
    if not sizes:
        sys.stderr.write(f"warning: {s.name}: no 1-GPU reference ({s.ref_name}) reports, "
                         "skipping\n")
        return False
    panels = s.panels()
    fig, axes = subplots(len(panels))
    for ax, panel in zip(axes, panels):
        ax.plot(sizes, [s.ref_mlups(n) for n in sizes], "--", color="#888888", lw=1.2,
                marker=".", label="Ideal (1-GPU reference)")
        has_runs = False
        for t, hp in s.curves(panel):
            pts = {n: rep.mlups_per_gpu for n in sizes
                   if (rep := s.get(panel, t, hp, n, g)) is not None}
            if pts:
                has_runs = True
                plot_curve(ax, pts, t, hp)
        pg = s.process_grid(panel, g) if panel[0] in sc.DISTRIBUTED_GRIDS else None
        ax.set_title(panel_title(panel) + (f", {pg}" if pg else ""))
        size_axis(ax, sizes)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="best", fontsize=8)
        if not has_runs:
            ax.text(0.5, 0.5, f"no runs at {g} GPUs", ha="center", transform=ax.transAxes)
    fig.suptitle(f"Carbon LBM MLUPS per GPU at {g} GPUs vs domain size\n"
                 f"{s.describe()}; ideal: 1-GPU {s.ref_name}", fontsize=10, y=1.02)
    fig.tight_layout()
    lr.save(fig, output, show)
    return True


def plot_hp_speedup(s: Series, out_dir: Path, show: bool) -> bool:
    import matplotlib.pyplot as plt

    if set(s.cfg["hp_halo"]) != {"off", "on"}:
        return False
    cmap = plt.get_cmap("viridis")
    sizes = s.cfg["sizes"]
    panels = s.panels()
    wrote = False
    for t in s.cfg["transports"]:
        tp = [p for p in panels if (t, "on") in s.curves(p)]
        if not tp:
            continue
        fig, axes = subplots(len(tp))
        for ax, panel in zip(axes, tp):
            for i, n in enumerate(sizes):
                xs, ys = [], []
                for g in s.gpus(panel):
                    on, off = s.get(panel, t, "on", n, g), s.get(panel, t, "off", n, g)
                    if on is not None and off is not None:
                        xs.append(g)
                        ys.append(on.mlups / off.mlups)
                if xs:
                    ax.plot(xs, ys, "D-", lw=1.5, ms=5, label=f"N={n}",
                            color=cmap(0.85 * i / max(1, len(sizes) - 1)))
            ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.7)
            ax.set_title(panel_title(panel))
            ax.grid(True, ls=":", alpha=0.5)
            ax.legend(loc="upper left", fontsize=8)
            gpu_axis(ax, s, panel, s.gpus(panel))
        for ax in axes[::min(3, len(tp))]:
            ax.set_ylabel("MLUPS(hp-halo on) / MLUPS(hp-halo off)")
        fig.suptitle(f"High-priority halo stream speedup, {lr.TRANSPORT_NAMES[t]}\n"
                     f"{s.describe()}", fontsize=10, y=1.02)
        fig.tight_layout()
        lr.save(fig, out_dir / f"{s.name}-hp-speedup-{t}.png", show)
        wrote = True
    return wrote


TRANSPORT_TAGS = {"intra": "P", "mpi": "N", "mpip2p": "M"}


def best_tag(grid: str, decomp: str, t: str, hp: str) -> str:
    tag = TRANSPORT_TAGS[t]
    if grid in sc.DISTRIBUTED_GRIDS:
        tag += decomp[0]
    return tag + ("+hp" if hp == "on" else "")


def plot_grids(s: Series, n: int, output: Path, show: bool) -> bool:
    import matplotlib.pyplot as plt

    grids = list(dict.fromkeys(grid for grid, *_ in s.cells))
    if len(grids) < 2:
        return False
    transports = s.cfg["transports"]
    t0 = "mpi" if "mpi" in transports else next((t for t in transports if t != "intra"),
                                                 "intra")
    hp0 = "off" if "off" in s.cfg["hp_halo"] else s.cfg["hp_halo"][0]
    ref = s.ref_mlups(n)
    g_max = s.max_gpus()
    fig = plt.figure(figsize=(12, 6.2))
    layout = fig.add_gridspec(2, 2, height_ratios=[4.0, 1.2], hspace=0.35)
    same = fig.add_subplot(layout[0, 0])
    best = fig.add_subplot(layout[0, 1], sharey=same)
    key = fig.add_subplot(layout[1, :])
    key.axis("off")
    has_runs = False
    tags: dict[str, dict[int, str]] = {}
    for grid in grids:
        color, marker = lr.GRID_STYLES[grid]
        style = dict(color=color, marker=marker, lw=1.8, ms=5, label=lr.GRID_TITLES[grid])
        panel = (grid, "slab")
        if (t0, hp0) in s.curves(panel):
            pts = {g: rep.mlups_per_gpu for g in s.gpus(panel)
                   if (rep := s.get(panel, t0, hp0, n, g)) is not None}
            if pts:
                xs = sorted(pts)
                same.plot(xs, [pts[g] for g in xs], **style)
                has_runs = True
        winners: dict[int, tuple[float, str]] = {}
        for gr, decomp, t, hp, g in s.cells:
            if gr != grid:
                continue
            rep = s.get((gr, decomp), t, hp, n, g)
            if rep is not None and rep.mlups_per_gpu > winners.get(g, (0.0, ""))[0]:
                winners[g] = (rep.mlups_per_gpu, best_tag(gr, decomp, t, hp))
        if winners:
            xs = sorted(winners)
            best.plot(xs, [winners[g][0] for g in xs], **style)
            tags[grid] = {g: winners[g][1] for g in xs}
            has_runs = True
    if not has_runs:
        plt.close(fig)
        return False
    for ax in (same, best):
        if ref is not None:
            ax.plot([1, g_max], [ref, ref], "--", color="#888888", lw=1.0,
                    label=f"Ideal (1 GPU: {ref:.0f})")
        ax.set_xticks(range(1, g_max + 1))
        ax.set_xlim(0.5, g_max + 0.5)
        ax.set_xlabel("Number of GPUs")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="lower left", fontsize=8)
    same.set_ylabel(lr.MLUPS_PER_GPU_LABEL + "\n(axis not from zero)")
    same.set_title(f"Same transport: {lr.TRANSPORT_NAMES[t0]}, slab"
                   + (", hp-halo on" if hp0 == "on" else ""))
    best.set_title("Fastest configuration per GPU count")
    plt.setp(best.get_yticklabels(), visible=False)
    gpus = list(range(1, g_max + 1))
    table = key.table(
        cellText=[[tags[grid].get(g, "") for g in gpus] for grid in tags],
        rowLabels=[lr.GRID_TITLES[grid] for grid in tags],
        rowColours=[lr.GRID_STYLES[grid][0] + "40" for grid in tags],
        colLabels=[f"{g} GPU" + ("s" if g > 1 else "") for g in gpus],
        cellLoc="center", bbox=[0.22, 0.12, 0.78, 0.88])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    key.text(0.0, 1.0, "Fastest configuration", fontsize=9, weight="bold", va="bottom",
             transform=key.transAxes)
    key.text(0.0, 0.0, "P Peeraccess, N NCCL, M MPI (P2P); s / p / b slab / pencil / brick "
             "(distributed grids); +hp hp-halo on", fontsize=8, color="#555555",
             transform=key.transAxes)
    fig.suptitle(f"Carbon LBM grids at N={n}: MLUPS per GPU\n{s.describe()}"
                 + (f"; ideal: 1-GPU {s.ref_name}" if ref is not None else ""),
                 fontsize=10)
    lr.save(fig, output, show)
    return True


def main() -> int:
    args = parse_args()
    try:
        cfg = sc.read_sweep_config(args.config)
        if args.sizes:
            cfg["sizes"] = args.sizes
        data_dir = lr.DATA_ROOT / cfg["prefix"]
        report_dir = lr.repo_dir(args.input_dir or data_dir / "json", must_exist=True)
        out_dir = lr.repo_dir(args.output_dir or data_dir, must_exist=False)
        ref_dir, ref_prefix, ref_label = sc.reference(cfg, report_dir)
        ref_dir = lr.repo_dir(ref_dir, must_exist=True)
    except (sc.SweepConfigError, ValueError) as e:
        sys.stderr.write(f"error: {e}\n")
        return 1
    plots = cfg["plots"]
    wrote = False
    for series, lattice, collision, precision in sc.series_prefixes(cfg):
        ref_series = f"{ref_prefix}-{lattice}-{collision}-{precision}"
        s = Series(series, report_dir, cfg, (ref_dir, ref_series, ref_label))
        if not s.reports:
            sys.stderr.write(f"warning: no reports for {series} in {report_dir}\n")
            continue
        if "vs_gpus" in plots:
            for n in cfg["sizes"]:
                wrote |= plot_vs_gpus(s, n, out_dir / f"{series}-mlups-per-gpu-vs-gpus-n{n}.png",
                                      args.show)
        if "vs_n" in plots:
            g = s.max_gpus()
            wrote |= plot_vs_n(s, g, out_dir / f"{series}-mlups-per-gpu-vs-n-g{g}.png",
                               args.show)
        if "hp_speedup" in plots:
            wrote |= plot_hp_speedup(s, out_dir, args.show)
        if "grids" in plots:
            for n in cfg["sizes"]:
                wrote |= plot_grids(s, n, out_dir / f"{series}-grids-n{n}.png", args.show)
    return 0 if wrote else 1


if __name__ == "__main__":
    sys.exit(main())
