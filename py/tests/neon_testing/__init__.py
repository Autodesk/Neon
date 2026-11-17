"""Shared helpers for the Neon Python test suite.

Importing this package first selects which ``neon`` to use (see ``env.py``),
so test modules must import it before ``neon`` and ``warp``.
"""

from .env import update_pythonpath

update_pythonpath()

from .runtime import (  # noqa: E402
    CONTAINER_RUNTIME,
    artifact_dir,
    export_vti_if_requested,
    gpu_count,
    init_once,
    mres_fp16_available,
    mres_fp16_container_available,
    non_primary_cuda_devices,
    require_gpu,
    require_gpus,
    run_container,
)
from .case import NeonTestCase  # noqa: E402
from . import grids, masks, operators  # noqa: E402
