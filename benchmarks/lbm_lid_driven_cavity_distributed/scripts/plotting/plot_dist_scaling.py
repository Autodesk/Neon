#!/usr/bin/env python3
"""Plot the distributed LBM sweep of ``lbm-dist-sweep-run.sh``.

Reads the sweep YAML and the JSON reports in ``data/<prefix>/json/`` and writes, for
every configs series (``<prefix>-<lattice>-<collision>-fp32``), one panel per
decomposition (slab | pencil | brick) and one curve per transport x hp-halo:

  <series>-eff-vs-gpus-n<N>.png    parallel efficiency vs GPU count, per size N
  <series>-mlups-vs-gpus-n<N>.png  MLUPS vs GPU count with the ideal line, per size N
  <series>-eff-vs-n-g<G>.png       parallel efficiency vs N at the largest GPU count G
  <series>-hp-speedup-<t>.png      MLUPS(hp-halo on) / MLUPS(hp-halo off) vs GPU count,
                                   one curve per N (only when hp_halo has off and on)

Efficiency vs GPUs is normalised to the 1-GPU ``ref`` run of the same N:

    eff(g) = MLUPS(curve, N, g) / (MLUPS(ref, N, 1) * g)

Efficiency vs N uses one constant baseline, the best 1-GPU ``ref`` over all sizes (as
in Neon Fig. 7), so its right axis is an exact GLUPS relabel. GPU tick labels carry the
process grid ProcessGrid chose (x*y*z), which shows where pencil / brick collapse to a
slab for prime GPU counts.

Example::

    python3 benchmarks/lbm_lid_driven_cavity_distributed/scripts/plotting/plot_dist_scaling.py \\
        --config benchmarks/lbm_lid_driven_cavity_distributed/scripts/lbm-dist-sweep.yml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA = "carbon-lbm-report-v1"
BENCH_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BENCH_ROOT.parents[1]
DATA_ROOT = BENCH_ROOT / "data"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lbm_dist_sweep_config  # noqa: E402

DECOMP_TITLES = {"slab": "Slab (z)", "pencil": "Pencil (y, z)", "brick": "Brick (x, y, z)"}
TRANSPORT_NAMES = {"mpi": "NCCL", "mpip2p": "MPI (P2P)"}
TRANSPORT_COLORS = {"mpi": "#1f77b4", "mpip2p": "#2ca02c"}
HP_STYLES = {"off": ("o", "--"), "on": ("D", "-")}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot the distributed LBM strong-scaling sweep.")
    p.add_argument("--config", type=Path, required=True,
                   help="Sweep YAML of lbm-dist-sweep-run.sh (e.g. scripts/lbm-dist-sweep.yml).")
    p.add_argument("--input-dir", type=Path, default=None,
                   help="Report directory inside the repo (default: data/<prefix>/json).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="PNG directory inside the repo (default: data/<prefix>/).")
    p.add_argument("--show", action="store_true", help="Pop each figure interactively.")
    return p.parse_args()


def repo_dir(path: Path, must_exist: bool) -> Path:
    """Resolve a command-line directory and keep it inside the carbon repo."""
    resolved = path.resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise ValueError(f"{resolved} is outside the repo {REPO_ROOT}")
    if must_exist and not resolved.is_dir():
        raise ValueError(f"not a directory: {resolved}")
    return resolved


def load_report(report_dir: Path, series: str, label: str, n: int, g: int) -> dict | None:
    path = report_dir / f"{series}-{label}-n{n}-g{g}.json"
    if not path.is_file():
        return None
    with path.open() as f:
        raw = json.load(f)
    if raw.get("schema") != SCHEMA:
        sys.stderr.write(f"warning: bad schema in {path.name}\n")
        return None
    return raw


def mlups_of(report: dict | None) -> float | None:
    return None if report is None else float(report["results"]["mlups"])


def curve_label(transport: str, hp: str) -> str:
    return TRANSPORT_NAMES[transport] + (" + hp-halo" if hp == "on" else "")


def curves(cfg: dict) -> list[tuple[str, str]]:
    return [(t, hp) for t in cfg["transports"] for hp in cfg["hp_halo"]]


class Series:
    """Every report of one configs series, indexed by (label, n, g)."""

    def __init__(self, report_dir: Path, series: str, cfg: dict) -> None:
        self.name = series
        self.cfg = cfg
        self.reports: dict[tuple[str, int, int], dict] = {}
        for n in cfg["sizes"]:
            ref = load_report(report_dir, series, "ref", n, 1)
            if ref is not None:
                self.reports[("ref", n, 1)] = ref
            for g in cfg["gpus"]:
                if g == 1:
                    continue
                for d in cfg["decompositions"]:
                    for t, hp in curves(cfg):
                        label = f"{d}-{t}-hp{hp}"
                        rep = load_report(report_dir, series, label, n, g)
                        if rep is not None:
                            self.reports[(label, n, g)] = rep

    def mlups(self, label: str, n: int, g: int) -> float | None:
        return mlups_of(self.reports.get((label, n, g)))

    def ref(self, n: int) -> float | None:
        return self.mlups("ref", n, 1)

    def curve(self, decomp: str, transport: str, hp: str, n: int) -> dict[int, float]:
        """MLUPS per GPU count, the 1-GPU point being the ref run."""
        pts: dict[int, float] = {}
        for g in self.cfg["gpus"]:
            m = self.ref(n) if g == 1 else self.mlups(f"{decomp}-{transport}-hp{hp}", n, g)
            if m is not None:
                pts[g] = m
        return pts

    def process_grid(self, decomp: str, g: int) -> str | None:
        if g == 1:
            return "1x1x1"
        for (label, _, gg), rep in self.reports.items():
            if gg == g and label.startswith(f"{decomp}-"):
                dims = rep.get("config", {}).get("process_grid")
                if dims:
                    return "x".join(str(v) for v in dims)
        return None

    def describe(self) -> str:
        rep = next(iter(self.reports.values()))
        cfg = rep.get("config", {})
        gpus = rep.get("system", {}).get("gpus", [])
        gpu = gpus[0]["name"] if gpus else "GPU"
        return (f"D3{str(cfg.get('lattice', 'q19')).upper()}+"
                f"{str(cfg.get('collision', 'bgk')).upper()} fp32, "
                f"overlap={cfg.get('overlap', '?')}, {gpu}")


def gpu_ticks(ax, s: Series, decomp: str, gpus: list[int]) -> None:
    ax.set_xticks(gpus)
    labels = []
    for g in gpus:
        pg = s.process_grid(decomp, g)
        labels.append(f"{g}\n{pg}" if pg else str(g))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Number of GPUs (process grid x*y*z)")


def plot_curve(ax, pts: dict[int, float], transport: str, hp: str, scale) -> None:
    xs = sorted(pts)
    marker, ls = HP_STYLES[hp]
    ax.plot(xs, [scale(g, pts[g]) for g in xs], marker=marker, ls=ls, lw=1.8, ms=5,
            color=TRANSPORT_COLORS[transport], label=curve_label(transport, hp))


def save(fig, path: Path, show: bool) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"plot saved to {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_vs_gpus(s: Series, n: int, out_dir: Path, show: bool) -> bool:
    import matplotlib.pyplot as plt

    ref = s.ref(n)
    if ref is None:
        sys.stderr.write(f"warning: {s.name}: no ref report for N={n}, skipping its plots\n")
        return False
    gpus = s.cfg["gpus"]
    decomps = s.cfg["decompositions"]
    g_max = max(gpus)

    fig, axes = plt.subplots(1, len(decomps), figsize=(5 * len(decomps), 4.5), sharey=True,
                             squeeze=False)
    for ax, d in zip(axes[0], decomps):
        for t, hp in curves(s.cfg):
            plot_curve(ax, s.curve(d, t, hp, n), t, hp, lambda g, m: m / (ref * g))
        ax.set_title(DECOMP_TITLES[d])
        ax.set_ylim(0.0, 1.05)
        ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="lower left", fontsize=9)
        gpu_ticks(ax, s, d, gpus)
    ideal_glups = ref * g_max / 1000.0
    secax = axes[0][-1].secondary_yaxis(
        "right", functions=(lambda e: e * ideal_glups, lambda v: v / ideal_glups))
    secax.set_ylabel(f"GLUPS (at {g_max} GPUs)")
    axes[0][0].set_ylabel("Parallel efficiency")
    fig.suptitle(f"Carbon distributed LBM strong scaling at N={n}\n"
                 f"{s.describe()}; reference: 1-GPU run ({ref:.0f} MLUPS)", fontsize=10, y=1.03)
    fig.tight_layout()
    save(fig, out_dir / f"{s.name}-eff-vs-gpus-n{n}.png", show)

    fig, axes = plt.subplots(1, len(decomps), figsize=(5 * len(decomps), 4.5), sharey=True,
                             squeeze=False)
    for ax, d in zip(axes[0], decomps):
        ax.plot([1, g_max], [ref, ref * g_max], "--", color="#888888", lw=1.0, alpha=0.7,
                label="Ideal linear")
        for t, hp in curves(s.cfg):
            plot_curve(ax, s.curve(d, t, hp, n), t, hp, lambda g, m: m)
        ax.set_title(DECOMP_TITLES[d])
        ax.set_ylim(bottom=0.0)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="upper left", fontsize=9)
        gpu_ticks(ax, s, d, gpus)
    axes[0][0].set_ylabel("MLUPS (10\u2076 lattice updates / s)")
    fig.suptitle(f"Carbon distributed LBM throughput vs GPU count at N={n}\n{s.describe()}",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    save(fig, out_dir / f"{s.name}-mlups-vs-gpus-n{n}.png", show)
    return True


def plot_vs_n(s: Series, out_dir: Path, show: bool) -> bool:
    import matplotlib.pyplot as plt

    g = max(s.cfg["gpus"])
    sizes = s.cfg["sizes"]
    refs = [r for r in (s.ref(n) for n in sizes) if r is not None]
    if g == 1 or not refs:
        return False
    baseline = max(refs)
    decomps = s.cfg["decompositions"]
    fig, axes = plt.subplots(1, len(decomps), figsize=(5 * len(decomps), 4.5), sharey=True,
                             squeeze=False)
    for ax, d in zip(axes[0], decomps):
        for t, hp in curves(s.cfg):
            pts = {n: m for n in sizes
                   if (m := s.mlups(f"{d}-{t}-hp{hp}", n, g)) is not None}
            plot_curve(ax, pts, t, hp, lambda _n, m: m / (baseline * g))
        pg = s.process_grid(d, g)
        ax.set_title(DECOMP_TITLES[d] + (f", {pg}" if pg else ""))
        ax.set_xlabel("Domain edge N")
        ax.set_xticks(sizes)
        ax.set_ylim(0.0, 1.05)
        ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="lower right", fontsize=9)
    ideal_glups = baseline * g / 1000.0
    secax = axes[0][-1].secondary_yaxis(
        "right", functions=(lambda e: e * ideal_glups, lambda v: v / ideal_glups))
    secax.set_ylabel("GLUPS")
    axes[0][0].set_ylabel("Parallel efficiency")
    fig.suptitle(f"Carbon distributed LBM parallel efficiency at {g} GPUs\n"
                 f"{s.describe()}; reference: best 1-GPU run ({baseline:.0f} MLUPS)",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    save(fig, out_dir / f"{s.name}-eff-vs-n-g{g}.png", show)
    return True


def plot_hp_speedup(s: Series, out_dir: Path, show: bool) -> bool:
    import matplotlib.pyplot as plt

    if set(s.cfg["hp_halo"]) != {"off", "on"}:
        return False
    gpus = [g for g in s.cfg["gpus"] if g > 1]
    if not gpus:
        return False
    decomps = s.cfg["decompositions"]
    cmap = plt.get_cmap("viridis")
    sizes = s.cfg["sizes"]
    wrote = False
    for t in s.cfg["transports"]:
        fig, axes = plt.subplots(1, len(decomps), figsize=(5 * len(decomps), 4.5), sharey=True,
                                 squeeze=False)
        for ax, d in zip(axes[0], decomps):
            for i, n in enumerate(sizes):
                xs, ys = [], []
                for g in gpus:
                    on = s.mlups(f"{d}-{t}-hpon", n, g)
                    off = s.mlups(f"{d}-{t}-hpoff", n, g)
                    if on is not None and off is not None:
                        xs.append(g)
                        ys.append(on / off)
                if xs:
                    ax.plot(xs, ys, "D-", lw=1.5, ms=5, label=f"N={n}",
                            color=cmap(0.85 * i / max(1, len(sizes) - 1)))
            ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.7)
            ax.set_title(DECOMP_TITLES[d])
            ax.grid(True, ls=":", alpha=0.5)
            ax.legend(loc="upper left", fontsize=8)
            gpu_ticks(ax, s, d, gpus)
        axes[0][0].set_ylabel("MLUPS(hp-halo on) / MLUPS(hp-halo off)")
        fig.suptitle(f"High-priority halo stream speedup, {TRANSPORT_NAMES[t]}\n{s.describe()}",
                     fontsize=10, y=1.03)
        fig.tight_layout()
        save(fig, out_dir / f"{s.name}-hp-speedup-{t}.png", show)
        wrote = True
    return wrote


def main() -> int:
    args = parse_args()
    try:
        cfg = lbm_dist_sweep_config.read_sweep_config(args.config)
        data_dir = DATA_ROOT / cfg["prefix"]
        report_dir = repo_dir(args.input_dir or data_dir / "json", must_exist=True)
        out_dir = repo_dir(args.output_dir or data_dir, must_exist=False)
    except (lbm_dist_sweep_config.SweepConfigError, ValueError) as e:
        sys.stderr.write(f"error: {e}\n")
        return 1

    wrote = False
    for series, *_ in lbm_dist_sweep_config.series_prefixes(cfg):
        s = Series(report_dir, series, cfg)
        if not s.reports:
            sys.stderr.write(f"warning: no reports for {series} in {report_dir}\n")
            continue
        for n in cfg["sizes"]:
            wrote |= plot_vs_gpus(s, n, out_dir, args.show)
        wrote |= plot_vs_n(s, out_dir, args.show)
        wrote |= plot_hp_speedup(s, out_dir, args.show)
    return 0 if wrote else 1


if __name__ == "__main__":
    sys.exit(main())
