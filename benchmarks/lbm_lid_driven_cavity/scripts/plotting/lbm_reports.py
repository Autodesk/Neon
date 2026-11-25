#!/usr/bin/env python3
"""Load ``carbon-lbm-report-v1`` JSON reports for the LBM plotters.

Reports live in ``<data_dir>/json/`` and are named ``<series>-<label>-n<N>-g<G>.json``
(see ``lbm-sweep-run.sh``) with ``<series> = <lattice>-<collision>-<precision>``. The
labels come from ``lbm_sweep_config.label``. Plots go to ``<data_dir>/plots/<experiment>/``.

Throughput per GPU is always ``results.mlups / G`` with G taken from the file name:
reports written under MPI before the driver counted every rank carry
``system.gpu_count = 1`` and a ``results.mlups_per_gpu`` equal to the total.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lbm_sweep_config as sc  # noqa: E402

SCHEMA = "carbon-lbm-report-v1"
BENCH_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BENCH_ROOT.parents[1]

GRID_TITLES = {
    "dense": "Dense (SoA)",
    "disg": "DenseDisg",
    "dense_distributed": "DenseDistributed",
    "disg_distributed": "DenseDisgDistributed",
}
GRID_STYLES = {
    "dense": ("#1f77b4", "o"),
    "disg": ("#ff7f0e", "o"),
    "dense_distributed": ("#2ca02c", "s"),
    "disg_distributed": ("#9467bd", "s"),
}
DECOMP_TITLES = {"slab": "slab (z)", "pencil": "pencil (y, z)", "brick": "brick (x, y, z)"}
TRANSPORT_NAMES = {"intra": "Peeraccess", "mpi": "NCCL", "mpip2p": "MPI (P2P)"}
TRANSPORT_COLORS = {"intra": "#d62728", "mpi": "#1f77b4", "mpip2p": "#2ca02c"}
HP_STYLES = {"off": ("D", "-"), "on": ("o", "--")}
MLUPS_PER_GPU_LABEL = "MLUPS per GPU (10\u2076 lattice updates / s / GPU)"


@dataclass(frozen=True)
class Report:
    path: Path
    gpus: int
    raw: dict

    @property
    def mlups(self) -> float:
        return float(self.raw["results"]["mlups"])

    @property
    def mlups_per_gpu(self) -> float:
        return self.mlups / self.gpus

    @property
    def process_grid(self) -> str | None:
        dims = self.raw.get("config", {}).get("process_grid")
        return "x".join(str(v) for v in dims) if dims else None


def repo_dir(path: Path, must_exist: bool) -> Path:
    """Resolve a command-line directory and keep it inside the carbon repo."""
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise ValueError(f"{resolved} is outside the repo {REPO_ROOT}")
    if must_exist and not resolved.is_dir():
        raise ValueError(f"not a directory: {resolved}")
    return resolved


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, required=True,
                   help="Sweep YAML of lbm-sweep-run.sh.")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Folder with json/ to read and plots/ to write (default: the "
                   "YAML's data_dir; e.g. benchmarks/lbm_lid_driven_cavity/reference).")
    p.add_argument("--only", default=None,
                   help="Comma-separated experiment names (default: every experiment).")
    p.add_argument("--show", action="store_true", help="Pop each figure interactively.")


def experiments(args: argparse.Namespace,
                figures: tuple[str, ...]) -> tuple[list[dict], Path, Path]:
    """``(experiments plotting any of figures, json dir, plots dir)`` for the parsed
    arguments; notes on stderr when there is none.

    Raises ``sc.SweepConfigError`` / ``ValueError`` on a bad YAML or folder.
    """
    cfg = sc.read_sweep_config(args.config, only=sc.split_names(args.only))
    base = repo_dir(args.data_dir or REPO_ROOT / cfg["data_dir"], must_exist=True)
    json_dir = repo_dir(base / "json", must_exist=True)
    exps = [e for e in cfg["experiments"].values() if set(e["plots"]) & set(figures)]
    if not exps:
        sys.stderr.write(f"note: no experiment of {args.config} plots {', '.join(figures)}\n")
    return exps, json_dir, base / "plots"


def load(report_dir: Path, series: str, label: str, n: int, g: int) -> Report | None:
    """The report of one run, or None when it is missing or not a v1 report."""
    path = Path(report_dir) / f"{series}-{label}-n{n}-g{g}.json"
    if not path.is_file():
        return None
    with path.open() as f:
        raw = json.load(f)
    if raw.get("schema") != SCHEMA:
        sys.stderr.write(f"warning: skipping {path.name}: bad schema\n")
        return None
    return Report(path, g, raw)


def describe(report: Report) -> str:
    """Run setup for figure titles, e.g. ``D3Q19+BGK fp32, overlap=standard, A100``.

    Reports written before ``--precision`` existed have no ``precision`` key; they are fp32.
    """
    cfg = report.raw.get("config", {})
    gpus = report.raw.get("system", {}).get("gpus", [])
    text = (f"D3{str(cfg.get('lattice', 'q19')).upper()}+"
            f"{str(cfg.get('collision', 'bgk')).upper()} {cfg.get('precision', 'fp32')}, "
            f"overlap={cfg.get('overlap', 'standard')}")
    return text + (f", {gpus[0]['name']}" if gpus else "")


def save(fig, path: Path, show: bool) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"plot saved to {path}")
    if show:
        plt.show()
    plt.close(fig)
