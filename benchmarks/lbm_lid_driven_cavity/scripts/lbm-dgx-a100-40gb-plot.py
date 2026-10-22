#!/usr/bin/env python3
"""Render the 8x A100-40GB Neon-Fig.7 parallel-efficiency figure.

Companion plotter for ``lbm-dgx-a100-40gb-run.sh``. It reads the JSON reports
that the run script wrote into ``data/lbm-dgx-a100-40gb/json/`` (every
artifact shares the ``lbm-dgx-a100-40gb`` prefix) and saves the two-panel
Peeraccess-vs-NCCL figure into ``data/lbm-dgx-a100-40gb/lbm-dgx-a100-40gb.png``
(the checked-in reference line).

Call the two back to back::

    ./benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-a100-40gb-run.sh
    python3 benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-a100-40gb-plot.py

The actual plotting is delegated to ``plot_neon_fig7.py`` (same directory) so
there is a single source of truth for the efficiency normalisation and axis
layout; this wrapper only pins the input/output defaults to ``data/``. Any
flag understood by ``plot_neon_fig7.py`` (e.g. ``--sizes``, ``--show``,
``--output``) can still be passed through and overrides the defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path

PREFIX = "lbm-dgx-a100-40gb"
HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parent / "data" / PREFIX
DATA_DIR = ROOT_DIR / "json"
DEFAULT_OUT = ROOT_DIR / f"{PREFIX}.png"

sys.path.insert(0, str(HERE))
import plot_neon_fig7 as fig7  # noqa: E402


def main() -> int:
    argv = sys.argv[1:]
    if not any(a == "--input-dir" or a.startswith("--input-dir=") for a in argv):
        sys.argv += ["--input-dir", str(DATA_DIR)]
    if not any(a == "--output" or a.startswith("--output=") for a in argv):
        sys.argv += ["--output", str(DEFAULT_OUT)]
    if not any(a == "--prefix" or a.startswith("--prefix=") for a in argv):
        sys.argv += ["--prefix", PREFIX]
    return fig7.main()


if __name__ == "__main__":
    sys.exit(main())
