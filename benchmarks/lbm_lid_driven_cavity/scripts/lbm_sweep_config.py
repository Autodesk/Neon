#!/usr/bin/env python3
"""Parse and validate the sweep YAML (see ``lbm-sweep.yml``).

Single source of truth for the keys, their defaults and their validation, shared by
``lbm-sweep-run.sh`` and the plotters in ``plotting/``. A sweep YAML holds

- global keys: how every run is timed and executed (``warmup``, ``iters_per_rep``,
  ``reps``, ``overlap``, ``runner``, ``data_dir``, ...);
- matrix keys at the top level: defaults for every experiment (``configs``, ``sizes``,
  ``grids``, ...);
- ``experiments``: a mapping ``name -> {type, <matrix keys>}``. The type decides which
  figures the plotters draw; the matrix keys override the top-level defaults.

All experiments share the timing, so a run two experiments ask for is executed once and
both read the same report ``<data_dir>/json/<lattice>-<collision>-<precision>-<label>-
n<N>-g<G>.json``.

- ``load_sweep_config(text, env, only)`` / ``read_sweep_config(path, only)`` return the
  global keys plus ``experiments``: ``name -> dict`` with every global and matrix key
  filled in, ``type`` and ``name``.
- ``runs(exp)`` enumerates the (grid, decomposition, transport, hp_halo, gpus) cells of one
  experiment, ``label(...)`` names their reports and ``all_runs(cfg)`` merges every
  experiment into the deduplicated list the run script executes.
- Run as a script it reads the YAML text from ``$LBM_SWEEP_CONFIG`` (experiments from
  ``$LBM_SWEEP_ONLY``, comma or space separated, empty = all), lets the env var of a global
  key (``FORCE``, ``SKIP_BUILD``, ...) override it, and prints ``VAR=value`` lines plus
  ``EXPERIMENTS=`` and ``RUNS=`` for the run script.

Values end up in shell variables, docker / mpiexec arguments and file names, so only
plain tokens (letters, digits, ``. _ : / -``) are accepted, and experiment names must be
single path components.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Global keys: (env var of lbm-sweep-run.sh, kind, default)
GLOBAL_KEYS: dict[str, tuple[str, str, object]] = {
    "runner": ("RUNNER", "word", "docker"),
    "warmup": ("WARMUP", "int", 400),
    "iters_per_rep": ("ITERS_PER_REP", "int", 1000),
    "reps": ("REPS", "int", 5),
    "overlap": ("OVERLAP", "word", "standard"),
    "reference_label": ("REFERENCE_LABEL", "name", "dense-intra"),
    "data_dir": ("DATA_DIR", "path", "benchmarks/lbm_lid_driven_cavity/data"),
    "sm_arch": ("SM_ARCH", "arch", "auto"),
    "image_intra": ("IMAGE_INTRA", "word", "carbon-cuda:local"),
    "image_mpi": ("IMAGE_MPI", "word", "carbon-cuda-nccl-pinned:local"),
    "build_dir": ("BUILD_DIR", "path", "build"),
    "build_mpi_dir": ("BUILD_MPI_DIR", "path", "build-mpi"),
    "skip_build": ("SKIP_BUILD", "bool", False),
    "force": ("FORCE", "bool", False),
    "run_timeout": ("RUN_TIMEOUT", "int", 1800),
}
# Matrix keys: (kind, default); allowed at the top level (defaults) and per experiment.
MATRIX_KEYS: dict[str, tuple[str, object]] = {
    "configs": ("words", ["q19:bgk"]),
    "precisions": ("words", ["fp32"]),
    "sizes": ("ints", list(range(256, 513, 64))),
    "gpus": ("ints", list(range(1, 9))),
    "grids": ("words", ["dense", "disg"]),
    "decompositions": ("words", ["slab"]),
    "gpus_pencil": ("ints", [4, 8]),
    "gpus_brick": ("ints", [8]),
    "transports": ("words", ["intra", "mpi", "mpip2p"]),
    "mpi_single_gpu": ("bool", False),
    "hp_halo": ("words", ["off"]),
    "plots": ("words", None),
}
# Experiment type -> the figures it can draw (all by default) and the plotter drawing them.
TYPES: dict[str, list[str]] = {
    "strong_scaling": ["vs_gpus", "hp_speedup", "grids", "disg_vs_dense"],
    "domain_size": ["vs_n"],
    "best_family": ["best_family"],
}
SINGLE_GRIDS = ("dense", "disg")
DISTRIBUTED_GRIDS = ("dense_distributed", "disg_distributed")
CHOICES = {
    "runner": {"docker", "local"},
    "configs": {"q19:bgk", "q27:bgk", "q27:kbc"},
    "precisions": {"fp32", "fp64"},
    "grids": set(SINGLE_GRIDS + DISTRIBUTED_GRIDS),
    "decompositions": {"slab", "pencil", "brick"},
    "transports": {"intra", "mpi", "mpip2p"},
    "hp_halo": {"off", "on"},
    "overlap": {"none", "standard", "extended", "twoway-extended"},
    "plots": {f for figs in TYPES.values() for f in figs},
}
TOKEN = re.compile(r"^[A-Za-z0-9._:/-]+$")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class SweepConfigError(ValueError):
    """Invalid sweep YAML."""


def _as_int(key: str, v: object) -> int:
    minimum = 0 if key == "warmup" else 1
    if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
        raise SweepConfigError(f"{key} must be an integer >= {minimum} (got {v!r})")
    return v


def _as_list(key: str, v: object) -> list:
    if isinstance(v, dict):
        if set(v) != {"start", "stop", "step"}:
            raise SweepConfigError(f"{key} range needs exactly start, stop, step")
        start, stop, step = (_as_int(key, v[k]) for k in ("start", "stop", "step"))
        return list(range(start, stop + 1, step))
    if isinstance(v, str):
        return v.split()
    if isinstance(v, list):
        return v
    return [v]


def _parse_value(key: str, kind: str, value: object) -> object:
    if kind == "int":
        return _as_int(key, value)
    if kind == "bool":
        if not isinstance(value, bool):
            raise SweepConfigError(f"{key} must be true or false (got {value!r})")
        return value
    if kind == "arch" and value == "auto":
        return "auto"
    if kind in ("ints", "arch"):
        items = [
            _as_int(key, int(x) if isinstance(x, str) and x.isdigit() else x)
            for x in _as_list(key, value)
        ]
        if not items:
            raise SweepConfigError(f"{key} is empty")
        return items
    items = _as_list(key, value) if kind == "words" else [value]
    if not items and key != "plots":
        raise SweepConfigError(f"{key} is empty")
    pattern = NAME if kind == "name" else TOKEN
    for x in items:
        if not isinstance(x, str) or not pattern.match(x):
            raise SweepConfigError(f"{key} has an invalid value {x!r}")
        if key in CHOICES and x not in CHOICES[key]:
            raise SweepConfigError(f"{key} value {x!r} not in {sorted(CHOICES[key])}")
        if kind == "path" and (x.startswith("/") or ".." in x.split("/")):
            raise SweepConfigError(f"{key} must be a path inside the carbon repo, relative to "
                                   f"its root, without '..' (got {x!r})")
    return items if kind == "words" else items[0]


def _env_value(kind: str, s: str) -> object:
    """An env var override as the YAML value it stands for (lists stay space-separated)."""
    if kind == "int":
        return int(s) if s.isdigit() else s
    if kind == "bool":
        return {"0": False, "1": True, "false": False, "true": True}.get(s.lower(), s)
    if kind == "arch" and s != "auto":
        return s.replace(";", " ")
    return s


def _parse_matrix(where: str, raw: dict) -> dict[str, object]:
    out = {}
    for key, value in raw.items():
        if key not in MATRIX_KEYS:
            raise SweepConfigError(f"{where}: unknown key {key!r} (matrix keys: "
                                   f"{', '.join(MATRIX_KEYS)})")
        out[key] = _parse_value(key, MATRIX_KEYS[key][0], value)
    return out


def load_sweep_config(text: str, env: dict[str, str] | None = None,
                      only: list[str] | None = None) -> dict[str, object]:
    """Validate YAML text; returns the global keys plus ``experiments``.

    ``env`` (e.g. ``os.environ``) overrides global keys: a key whose env var
    (``GLOBAL_KEYS``) is set takes that value, in the run script's format (0/1 bools).
    ``only`` keeps the named experiments (all when empty).
    """
    try:
        import yaml
    except ImportError:
        raise SweepConfigError("PyYAML is not installed: pip install pyyaml "
                               "(or apt install python3-yaml)") from None

    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise SweepConfigError("top level must be a mapping")
    experiments_raw = raw.pop("experiments", None)
    if not isinstance(experiments_raw, dict) or not experiments_raw:
        raise SweepConfigError("experiments must be a non-empty mapping name -> settings")
    global_raw = {k: v for k, v in raw.items() if k in GLOBAL_KEYS}
    defaults = _parse_matrix("top level", {k: v for k, v in raw.items()
                                           if k not in GLOBAL_KEYS})
    for key, (var, kind, _) in GLOBAL_KEYS.items():
        if env is not None and var in env:
            global_raw[key] = _env_value(kind, env[var])
    cfg: dict[str, object] = {key: default for key, (_, _, default) in GLOBAL_KEYS.items()}
    for key, value in global_raw.items():
        cfg[key] = _parse_value(key, GLOBAL_KEYS[key][1], value)

    experiments: dict[str, dict[str, object]] = {}
    for name, body in experiments_raw.items():
        if not isinstance(name, str) or not NAME.match(name):
            raise SweepConfigError(f"invalid experiment name {name!r} (letters, digits, . _ -)")
        if not isinstance(body, dict) or "type" not in body:
            raise SweepConfigError(f"experiment {name}: needs a type ({', '.join(TYPES)})")
        body = dict(body)
        etype = body.pop("type")
        if etype not in TYPES:
            raise SweepConfigError(f"experiment {name}: type {etype!r} not in {list(TYPES)}")
        exp = {key: default for key, (_, default) in MATRIX_KEYS.items()}
        exp.update(defaults)
        exp.update(_parse_matrix(f"experiment {name}", body))
        exp.update({k: cfg[k] for k in GLOBAL_KEYS})
        exp["name"] = name
        exp["type"] = etype
        if exp["plots"] is None:
            exp["plots"] = list(TYPES[etype])
        validate(exp)
        experiments[name] = exp
    if only:
        unknown = [n for n in only if n not in experiments]
        if unknown:
            raise SweepConfigError(f"unknown experiment(s) {unknown} (have {list(experiments)})")
        experiments = {n: e for n, e in experiments.items() if n in only}
    cfg["experiments"] = experiments
    return cfg


def validate(exp: dict[str, object]) -> None:
    """Cross-key checks of one experiment; raises ``SweepConfigError``."""
    name = exp["name"]
    bad = [f for f in exp["plots"] if f not in TYPES[exp["type"]]]
    if bad:
        raise SweepConfigError(f"experiment {name}: plots {bad} do not belong to type "
                               f"{exp['type']} ({TYPES[exp['type']]})")
    for key in ("gpus_pencil", "gpus_brick"):
        if any(g < 2 for g in exp[key]):
            raise SweepConfigError(f"experiment {name}: {key} entries must be >= 2 (a 1-rank "
                                   "run has no decomposition)")
    distributed = [g for g in exp["grids"] if g in DISTRIBUTED_GRIDS]
    if distributed and not set(exp["transports"]) & {"mpi", "mpip2p"}:
        raise SweepConfigError(f"experiment {name}: grids {distributed} run under mpiexec "
                               "only; add mpi or mpip2p to transports")
    if "on" in exp["hp_halo"] and exp["overlap"] == "none":
        raise SweepConfigError(f"experiment {name}: hp_halo on needs overlap != none (no halo "
                               "stream to prioritise otherwise)")
    if exp["type"] == "best_family" and not ({"dense", "disg"} <= set(exp["grids"])
                                             and {"intra", "mpi"} <= set(exp["transports"])):
        raise SweepConfigError(f"experiment {name}: best_family needs grids dense and disg "
                               "with transports intra and mpi")
    if not runs(exp):
        raise SweepConfigError(f"experiment {name}: the grids x transports x gpus matrix is "
                               "empty")


def gpus_of(exp: dict[str, object], decomposition: str) -> list[int]:
    """GPU counts swept for a decomposition: slab uses ``gpus``, the others their own key."""
    return exp["gpus"] if decomposition == "slab" else exp[f"gpus_{decomposition}"]


def runs(exp: dict[str, object]) -> list[tuple[str, str, str, str, int]]:
    """``(grid, decomposition, transport, hp_halo, gpus)`` of every run, sizes aside.

    - ``dense`` / ``disg`` run on every ``gpus`` entry with every transport; decomposition
      is always ``slab`` (single-process grids split z across GPUs).
    - ``dense_distributed`` / ``disg_distributed`` need one MPI rank per GPU, so they skip
      ``intra``; each decomposition runs on its own GPU counts (``gpus_of``).
    - ``mpi`` / ``mpip2p`` skip G=1: a single rank has no halo exchange. With
      ``mpi_single_gpu`` the single-process grids run them at G=1 anyway (one rank,
      pinned to GPU 0 by the run script), so per-transport plots start at 1 GPU.
    """
    out = []
    for grid in exp["grids"]:
        distributed = grid in DISTRIBUTED_GRIDS
        decomps = exp["decompositions"] if distributed else ["slab"]
        for decomp in decomps:
            gpus = gpus_of(exp, decomp) if distributed else exp["gpus"]
            for transport in exp["transports"]:
                if distributed and transport == "intra":
                    continue
                for hp in exp["hp_halo"]:
                    for g in gpus:
                        if (transport != "intra" and g == 1
                                and (distributed or not exp["mpi_single_gpu"])):
                            continue
                        out.append((grid, decomp, transport, hp, g))
    return out


def label(grid: str, decomposition: str, transport: str, hp_halo: str) -> str:
    """Report label of one run, the ``<label>`` in ``<series>-<label>-n<N>-g<G>.json``.

    Single-process grids keep the historic ``<grid>-<transport>`` (``-hpon`` appended
    when the halo stream is prioritised); distributed grids spell out everything:
    ``<grid>-<decomposition>-<transport>-hp<off|on>``.
    """
    if grid in DISTRIBUTED_GRIDS:
        return f"{grid}-{decomposition}-{transport}-hp{hp_halo}"
    return f"{grid}-{transport}" + ("-hpon" if hp_halo == "on" else "")


def series_prefixes(exp: dict[str, object]) -> list[tuple[str, str, str, str]]:
    """``(series, lattice, collision, precision)`` for every configs x precisions entry.

    A series (``<lattice>-<collision>-<precision>``) is what the run script puts in front
    of ``<label>-n<N>-g<G>.json``.
    """
    out = []
    for c in exp["configs"]:
        lattice, collision = c.split(":")
        for precision in exp["precisions"]:
            out.append((f"{lattice}-{collision}-{precision}", lattice, collision, precision))
    return out


def all_runs(cfg: dict[str, object]) -> list[tuple[str, ...]]:
    """Every run of every selected experiment, once: ``(lattice, collision, precision,
    grid, decomposition, transport, hp_halo, gpus, n, label)``, in experiment order."""
    seen: dict[tuple[str, ...], None] = {}
    for exp in cfg["experiments"].values():
        for _, lattice, collision, precision in series_prefixes(exp):
            for grid, decomp, t, hp, g in runs(exp):
                for n in exp["sizes"]:
                    seen.setdefault((lattice, collision, precision, grid, decomp, t, hp,
                                     str(g), str(n), label(grid, decomp, t, hp)), None)
    return list(seen)


def experiment_run_count(exp: dict[str, object]) -> int:
    return len(series_prefixes(exp)) * len(runs(exp)) * len(exp["sizes"])


def read_sweep_config(path: Path, only: list[str] | None = None) -> dict[str, object]:
    """Load a sweep YAML file; the path must be an existing ``.yml`` / ``.yaml`` file."""
    path = Path(path).resolve()
    if path.suffix not in (".yml", ".yaml") or not path.is_file():
        raise SweepConfigError(f"not a .yml/.yaml file: {path}")
    return load_sweep_config(path.read_text(), only=only)


def split_names(s: str | None) -> list[str]:
    return [x for x in re.split(r"[,\s]+", s or "") if x]


def _shell_value(kind: str, value: object) -> str:
    if kind == "bool":
        return "1" if value else "0"
    if kind == "arch" and value != "auto":
        return ";".join(map(str, value))
    return str(value)


def main() -> int:
    try:
        cfg = load_sweep_config(os.environ.get("LBM_SWEEP_CONFIG", ""), dict(os.environ),
                                split_names(os.environ.get("LBM_SWEEP_ONLY")))
    except SweepConfigError as e:
        sys.stderr.write(f"error: config: {e}\n")
        return 1
    for key, (var, kind, _) in GLOBAL_KEYS.items():
        if var in os.environ:
            sys.stderr.write(f"==> {var} taken from the environment\n")
        print(f"{var}={_shell_value(kind, cfg[key])}")
    print("EXPERIMENTS=" + " ".join(f"{e['name']}:{e['type']}:{experiment_run_count(e)}"
                                    for e in cfg["experiments"].values()))
    print("RUNS=" + " ".join(":".join(r) for r in all_runs(cfg)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
