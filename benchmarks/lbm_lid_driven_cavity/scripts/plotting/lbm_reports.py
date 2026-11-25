#!/usr/bin/env python3
"""Load ``carbon-lbm-report-v1`` JSON reports for the LBM plotters.

Reports are named ``<series>-<label>-n<N>-g<G>.json`` (see ``lbm-dgx-sweep-run.sh``)
with ``<series> = <prefix>-<lattice>-<collision>-<precision>``. The labels come from
``lbm_sweep_config.label``.

Throughput per GPU is always ``results.mlups / G`` with G taken from the file name:
reports written under MPI before the driver counted every rank carry
``system.gpu_count = 1`` and a ``results.mlups_per_gpu`` equal to the total.

Distributed reports of the retired ``lbm-dist-sweep-run.sh`` used other names; ``load``
falls back to them, so older result sets still plot:

  <series>-<decomposition>-<transport>-hp<off|on>          (one grid swept)
  <prefix>-<grid>-<lattice>-<collision>-<precision>-<...>  (several grids swept)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "carbon-lbm-report-v1"
BENCH_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BENCH_ROOT.parents[1]
DATA_ROOT = BENCH_ROOT / "data"

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


def _legacy_names(series: str, label: str) -> list[str]:
    parts = label.split("-")
    if len(parts) != 4 or not parts[0].endswith("_distributed"):
        return []
    grid, rest = parts[0], "-".join(parts[1:])
    names = [f"{series}-{rest}"]
    head = series.rsplit("-", 3)
    if len(head) == 4:
        names.append(f"{head[0]}-{grid}-{'-'.join(head[1:])}-{rest}")
    return names


def load(report_dir: Path, series: str, label: str, n: int, g: int) -> Report | None:
    """The report of one run, or None when it is missing or not a v1 report."""
    for stem in [f"{series}-{label}"] + _legacy_names(series, label):
        path = Path(report_dir) / f"{stem}-n{n}-g{g}.json"
        if path.is_file():
            break
    else:
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
