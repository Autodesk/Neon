#!/usr/bin/env python3
"""Parse and validate the sweep YAML (see ``lbm-dgx-sweep.yml``).

Single source of truth for the keys, their defaults and their validation, shared by
``lbm-dgx-sweep-run.sh`` and the plotters in ``plotting/``:

- ``load_sweep_config(text)`` / ``read_sweep_config(path)`` return every key as a Python
  value (lists of ints or strings, ints, bools, strings), defaults filled in.
- Run as a script, it reads the YAML text from ``$LBM_SWEEP_CONFIG`` (empty = all
  defaults) and prints ``VAR=value`` lines with the run script's env var names.

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
    "transports": ("TRANSPORTS", "words", ["intra", "mpi", "mpip2p"]),
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
CHOICES = {
    "configs": {"q19:bgk", "q27:bgk", "q27:kbc"},
    "precisions": {"fp32", "fp64"},
    "grids": {"dense", "disg"},
    "transports": {"intra", "mpi", "mpip2p"},
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


def load_sweep_config(
    text: str,
    keys: dict[str, tuple[str, str, object]] = KEYS,
    choices: dict[str, set[str]] = CHOICES,
) -> dict[str, object]:
    """Validate YAML text; returns every key in ``keys``, omitted ones at their default.

    ``keys`` / ``choices`` default to this sweep's tables; other sweeps (e.g. the
    distributed benchmark's) pass their own in the same format.
    """
    import yaml

    cfg = yaml.safe_load(text) or {}
    if not isinstance(cfg, dict):
        raise SweepConfigError("top level must be a mapping")
    out = {key: default for key, (_, _, default) in keys.items()}
    for key, value in cfg.items():
        if key not in keys:
            raise SweepConfigError(f"unknown key {key!r} (known: {', '.join(keys)})")
        out[key] = _parse_value(key, keys[key][1], value, choices)
    return out


def read_sweep_config(
    path: Path,
    keys: dict[str, tuple[str, str, object]] = KEYS,
    choices: dict[str, set[str]] = CHOICES,
) -> dict[str, object]:
    """Load a sweep YAML file; the path must be an existing ``.yml`` / ``.yaml`` file."""
    path = Path(path).resolve()
    if path.suffix not in (".yml", ".yaml") or not path.is_file():
        raise SweepConfigError(f"not a .yml/.yaml file: {path}")
    return load_sweep_config(path.read_text(), keys, choices)


def series_prefixes(cfg: dict[str, object]) -> list[tuple[str, str, str, str]]:
    """``(series_prefix, lattice, collision, precision)`` for every configs x precisions entry.

    A series prefix (``<prefix>-<lattice>-<collision>-<precision>``) is what the run script
    puts in front of ``<grid>-<transport>-n<N>-g<G>.json``.
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


def main(
    keys: dict[str, tuple[str, str, object]] = KEYS,
    choices: dict[str, set[str]] = CHOICES,
) -> int:
    try:
        cfg = load_sweep_config(os.environ.get("LBM_SWEEP_CONFIG", ""), keys, choices)
    except SweepConfigError as e:
        sys.stderr.write(f"error: config: {e}\n")
        return 1
    for key, (var, kind, _) in keys.items():
        print(f"{var}={_shell_value(kind, cfg[key])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
