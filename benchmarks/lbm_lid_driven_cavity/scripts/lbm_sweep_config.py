#!/usr/bin/env python3
"""Parse and validate the sweep YAML (see ``lbm-dgx-sweep.yml``).

Single source of truth for the keys, their defaults and their validation, shared by
``lbm-dgx-sweep-run.sh`` and the plotters in ``plotting/``:

- ``load_sweep_config(text)`` / ``read_sweep_config(path)`` return every key as a Python
  value (lists of ints or strings, ints, bools, strings), defaults filled in.
- Run as a script, it reads the YAML text from ``$LBM_SWEEP_CONFIG`` (empty = all
  defaults), lets a set env var of a key (``CONFIGS``, ``GRIDS``, ...) override it, and
  prints ``VAR=value`` lines with the run script's env var names plus ``RUNS=``.
- ``runs(cfg)`` enumerates the (grid, decomposition, transport, hp_halo, gpus) cells the
  run script executes, and ``label(...)`` names their reports; the plotters use both.

Values end up in shell variables, docker / mpiexec arguments and file names, so only
plain tokens (letters, digits, ``. _ : / -``) are accepted, and ``prefix`` must be a
single path component.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# key: (env var of lbm-dgx-sweep-run.sh, kind, default)
KEYS: dict[str, tuple[str, str, object]] = {
    "configs": ("CONFIGS", "words", ["q19:bgk"]),
    "precisions": ("PRECISIONS", "words", ["fp32"]),
    "sizes": ("SIZES", "ints", list(range(256, 513, 64))),
    "gpus": ("GPUS_LIST", "ints", list(range(1, 9))),
    "grids": ("GRIDS", "words", ["dense", "disg"]),
    "decompositions": ("DECOMPOSITIONS", "words", ["slab"]),
    "gpus_pencil": ("GPUS_PENCIL", "ints", [4, 8]),
    "gpus_brick": ("GPUS_BRICK", "ints", [8]),
    "transports": ("TRANSPORTS", "words", ["intra", "mpi", "mpip2p"]),
    "mpi_single_gpu": ("MPI_SINGLE_GPU", "bool", False),
    "hp_halo": ("HP_HALO", "words", ["off"]),
    "overlap": ("OVERLAP", "word", "standard"),
    "reference_dir": ("REFERENCE_DIR", "optword", ""),
    "reference_prefix": ("REFERENCE_PREFIX", "optname", ""),
    "reference_label": ("REFERENCE_LABEL", "name", "dense-intra"),
    "plots": ("PLOTS", "words", ["vs_gpus", "vs_n", "hp_speedup", "grids"]),
    "warmup": ("WARMUP", "int", 400),
    "iters_per_rep": ("ITERS_PER_REP", "int", 400),
    "reps": ("REPS", "int", 4),
    "sm_arch": ("SM_ARCH", "arch", "auto"),
    "image_intra": ("IMAGE_INTRA", "word", "carbon-cuda:local"),
    "image_mpi": ("IMAGE_MPI", "word", "carbon-cuda-nccl-pinned:local"),
    "prefix": ("PREFIX", "name", "lbm-dgx-sweep"),
    "skip_build": ("SKIP_BUILD", "bool", False),
    "force": ("FORCE", "bool", False),
    "run_timeout": ("RUN_TIMEOUT", "int", 1800),
}
SINGLE_GRIDS = ("dense", "disg")
DISTRIBUTED_GRIDS = ("dense_distributed", "disg_distributed")
CHOICES = {
    "configs": {"q19:bgk", "q27:bgk", "q27:kbc"},
    "precisions": {"fp32", "fp64"},
    "grids": set(SINGLE_GRIDS + DISTRIBUTED_GRIDS),
    "decompositions": {"slab", "pencil", "brick"},
    "transports": {"intra", "mpi", "mpip2p"},
    "hp_halo": {"off", "on"},
    "overlap": {"none", "standard", "extended", "twoway-extended"},
    "plots": {"vs_gpus", "vs_n", "hp_speedup", "grids"},
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


def _parse_value(key: str, kind: str, value: object, choices: dict[str, set[str]]) -> object:
    if kind.startswith("opt"):
        if value is None or value == "":
            return ""
        kind = kind[3:]
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
    if not items:
        raise SweepConfigError(f"{key} is empty")
    pattern = NAME if kind == "name" else TOKEN
    for x in items:
        if not isinstance(x, str) or not pattern.match(x):
            raise SweepConfigError(f"{key} has an invalid value {x!r}")
        if key in choices and x not in choices[key]:
            raise SweepConfigError(f"{key} value {x!r} not in {sorted(choices[key])}")
    return items if kind == "words" else items[0]


def _env_value(kind: str, s: str) -> object:
    """An env var override as the YAML value it stands for (lists stay space-separated)."""
    kind = kind.removeprefix("opt")
    if kind == "int":
        return int(s) if s.isdigit() else s
    if kind == "bool":
        return {"0": False, "1": True, "false": False, "true": True}.get(s.lower(), s)
    if kind == "arch" and s != "auto":
        return s.replace(";", " ")
    return s


def load_sweep_config(text: str, env: dict[str, str] | None = None) -> dict[str, object]:
    """Validate YAML text; returns every key in ``KEYS``, omitted ones at their default.

    ``env`` (e.g. ``os.environ``) overrides the YAML: a key whose env var (``KEYS``) is set
    takes that value, in the run script's format (space-separated lists, 0/1 bools).
    """
    import yaml

    cfg = yaml.safe_load(text) or {}
    if not isinstance(cfg, dict):
        raise SweepConfigError("top level must be a mapping")
    for key in cfg:
        if key not in KEYS:
            raise SweepConfigError(f"unknown key {key!r} (known: {', '.join(KEYS)})")
    for key, (var, kind, _) in KEYS.items():
        if env is not None and var in env:
            cfg[key] = _env_value(kind, env[var])
    out = {key: default for key, (_, _, default) in KEYS.items()}
    for key, value in cfg.items():
        out[key] = _parse_value(key, KEYS[key][1], value, CHOICES)
    validate(out)
    return out


def validate(cfg: dict[str, object]) -> None:
    """Cross-key checks of a loaded config; raises ``SweepConfigError``."""
    for key in ("gpus_pencil", "gpus_brick"):
        if any(g < 2 for g in cfg[key]):
            raise SweepConfigError(f"{key} entries must be >= 2 (a 1-rank run has no "
                                   "decomposition)")
    distributed = [g for g in cfg["grids"] if g in DISTRIBUTED_GRIDS]
    if distributed and not set(cfg["transports"]) & {"mpi", "mpip2p"}:
        raise SweepConfigError(f"grids {distributed} run under mpiexec only; add mpi or "
                               "mpip2p to transports")
    if "on" in cfg["hp_halo"] and cfg["overlap"] == "none":
        raise SweepConfigError("hp_halo on needs overlap != none (no halo stream to "
                               "prioritise otherwise)")
    if not runs(cfg):
        raise SweepConfigError("the grids x transports x gpus matrix is empty")


def gpus_of(cfg: dict[str, object], decomposition: str) -> list[int]:
    """GPU counts swept for a decomposition: slab uses ``gpus``, the others their own key."""
    return cfg["gpus"] if decomposition == "slab" else cfg[f"gpus_{decomposition}"]


def runs(cfg: dict[str, object]) -> list[tuple[str, str, str, str, int]]:
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
    for grid in cfg["grids"]:
        distributed = grid in DISTRIBUTED_GRIDS
        decomps = cfg["decompositions"] if distributed else ["slab"]
        for decomp in decomps:
            gpus = gpus_of(cfg, decomp) if distributed else cfg["gpus"]
            for transport in cfg["transports"]:
                if distributed and transport == "intra":
                    continue
                for hp in cfg["hp_halo"]:
                    for g in gpus:
                        if (transport != "intra" and g == 1
                                and (distributed or not cfg["mpi_single_gpu"])):
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


def reference(cfg: dict[str, object], sweep_dir: Path) -> tuple[Path, str, str]:
    """``(json_dir, prefix, label)`` of the 1-GPU MLUPS/GPU reference runs.

    The reference of series ``<prefix>-<tail>`` at size N is
    ``<json_dir>/<ref_prefix>-<tail>-<label>-n<N>-g1.json``. An empty ``reference_dir``
    means the sweep's own reports (``sweep_dir``); a relative one is relative to the carbon
    repo root.
    """
    if cfg["reference_dir"]:
        ref_dir = Path(cfg["reference_dir"])
        if not ref_dir.is_absolute():
            ref_dir = Path(__file__).resolve().parents[3] / ref_dir
    else:
        ref_dir = Path(sweep_dir)
    return ref_dir, cfg["reference_prefix"] or cfg["prefix"], cfg["reference_label"]


def read_sweep_config(path: Path) -> dict[str, object]:
    """Load a sweep YAML file; the path must be an existing ``.yml`` / ``.yaml`` file."""
    path = Path(path).resolve()
    if path.suffix not in (".yml", ".yaml") or not path.is_file():
        raise SweepConfigError(f"not a .yml/.yaml file: {path}")
    return load_sweep_config(path.read_text())


def series_prefixes(cfg: dict[str, object]) -> list[tuple[str, str, str, str]]:
    """``(series_prefix, lattice, collision, precision)`` for every configs x precisions entry.

    A series prefix (``<prefix>-<lattice>-<collision>-<precision>``) is what the run script
    puts in front of ``<label>-n<N>-g<G>.json``.
    """
    out = []
    for c in cfg["configs"]:
        lattice, collision = c.split(":")
        for precision in cfg["precisions"]:
            out.append((f"{cfg['prefix']}-{lattice}-{collision}-{precision}", lattice, collision,
                        precision))
    return out


def _shell_value(kind: str, value: object) -> str:
    if kind == "bool":
        return "1" if value else "0"
    if kind == "arch" and value != "auto":
        return ";".join(map(str, value))
    if isinstance(value, list):
        return " ".join(map(str, value))
    return str(value)


def main() -> int:
    try:
        cfg = load_sweep_config(os.environ.get("LBM_SWEEP_CONFIG", ""), dict(os.environ))
    except SweepConfigError as e:
        sys.stderr.write(f"error: config: {e}\n")
        return 1
    for key, (var, kind, _) in KEYS.items():
        if var in os.environ:
            sys.stderr.write(f"==> {var} taken from the environment\n")
        print(f"{var}={_shell_value(kind, cfg[key])}")
    # One <grid>:<decomposition>:<transport>:<hp>:<G>:<label> token per run.
    print("RUNS=" + " ".join(f"{gr}:{d}:{t}:{hp}:{g}:{label(gr, d, t, hp)}"
                             for gr, d, t, hp, g in runs(cfg)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
