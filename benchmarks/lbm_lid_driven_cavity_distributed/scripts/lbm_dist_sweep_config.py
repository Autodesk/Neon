#!/usr/bin/env python3
"""Parse and validate the distributed sweep YAML (see ``lbm-dist-sweep.yml``).

Keys, defaults and validation shared by ``lbm-dist-sweep-run.sh`` and
``plotting/plot_dist_scaling.py``. Parsing itself is the single-process sweep's
``lbm_sweep_config`` with this sweep's tables:

- ``read_sweep_config(path)`` returns every key as a Python value, defaults filled in.
- Run as a script, it reads the YAML text from ``$LBM_SWEEP_CONFIG`` (empty = all
  defaults) and prints ``VAR=value`` lines with the run script's env var names.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "lbm_lid_driven_cavity" / "scripts")
)
import lbm_sweep_config  # noqa: E402

SweepConfigError = lbm_sweep_config.SweepConfigError

# The benchmark's populations are fp32 only; report names carry it so they line up
# with the single-process sweep's.
PRECISION = "fp32"

# key: (env var of lbm-dist-sweep-run.sh, kind, default)
KEYS: dict[str, tuple[str, str, object]] = {
    "configs": ("CONFIGS", "words", ["q19:bgk"]),
    "sizes": ("SIZES", "ints", list(range(256, 513, 64))),
    "decompositions": ("DECOMPOSITIONS", "words", ["slab", "pencil", "brick"]),
    "gpus_slab": ("GPUS_SLAB", "ints", list(range(2, 9))),
    "gpus_pencil": ("GPUS_PENCIL", "ints", [4, 8]),
    "gpus_brick": ("GPUS_BRICK", "ints", [8]),
    "transports": ("TRANSPORTS", "words", ["mpi"]),
    "hp_halo": ("HP_HALO", "words", ["off", "on"]),
    "overlap": ("OVERLAP", "word", "standard"),
    "reference_dir": ("REFERENCE_DIR", "word",
                      "benchmarks/lbm_lid_driven_cavity/data/lbm-dgx-sweep/json"),
    "reference_prefix": ("REFERENCE_PREFIX", "name", "lbm-dgx-sweep"),
    "reference_label": ("REFERENCE_LABEL", "name", "dense-intra"),
    "warmup": ("WARMUP", "int", 400),
    "iters_per_rep": ("ITERS_PER_REP", "int", 400),
    "reps": ("REPS", "int", 4),
    "sm_arch": ("SM_ARCH", "arch", "auto"),
    "image": ("IMAGE", "word", "carbon-cuda-nccl-pinned:local"),
    "build_dir": ("BUILD_DIR", "name", "build-dist"),
    "prefix": ("PREFIX", "name", "lbm-dist-sweep"),
    "skip_build": ("SKIP_BUILD", "bool", False),
    "force": ("FORCE", "bool", False),
    "run_timeout": ("RUN_TIMEOUT", "int", 1800),
}
CHOICES = {
    "configs": {"q19:bgk", "q27:bgk", "q27:kbc"},
    "decompositions": {"slab", "pencil", "brick"},
    "transports": {"mpi", "mpip2p"},
    "hp_halo": {"off", "on"},
    "overlap": {"none", "standard", "extended", "twoway-extended"},
}


def read_sweep_config(path: Path) -> dict[str, object]:
    """Load a distributed sweep YAML file; the path must be an existing ``.yml`` / ``.yaml``."""
    cfg = lbm_sweep_config.read_sweep_config(path, KEYS, CHOICES)
    for d in CHOICES["decompositions"]:
        if any(g < 2 for g in cfg[f"gpus_{d}"]):
            raise SweepConfigError(f"gpus_{d} entries must be >= 2 (the 1-GPU point is the "
                                   "reference run)")
    return cfg


def gpus_of(cfg: dict[str, object], decomposition: str) -> list[int]:
    return cfg[f"gpus_{decomposition}"]


def series_prefixes(cfg: dict[str, object]) -> list[tuple[str, str, str, str]]:
    """``(series_prefix, reference_series, lattice, collision)`` for every configs entry.

    A series prefix (``<prefix>-<lattice>-<collision>-fp32``) is what the run script puts
    in front of ``<decomposition>-<transport>-hp<off|on>-n<N>-g<G>.json``. The efficiency
    baseline of size N is ``<reference_dir>/<reference_series>-<reference_label>-n<N>-g1.json``
    with ``reference_series = <reference_prefix>-<lattice>-<collision>-fp32``, by default the
    1-GPU Dense run of the single-process DGX sweep.
    """
    out = []
    for c in cfg["configs"]:
        lattice, collision = c.split(":")
        tail = f"{lattice}-{collision}-{PRECISION}"
        out.append((f"{cfg['prefix']}-{tail}", f"{cfg['reference_prefix']}-{tail}", lattice,
                    collision))
    return out


if __name__ == "__main__":
    sys.exit(lbm_sweep_config.main(KEYS, CHOICES))
