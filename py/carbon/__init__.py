"""Carbon Python / NVIDIA Warp bindings (CUDA-only)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .gate import Gate
from .index3d import Index_3d
from .system import System
from .dense import Dense, DenseField
from .block import Block, BlockField
from .multires import Mgrid, MgridField
from .kernel import BoundKernel, block_kernel, build_bound_kernel, dense_kernel, kernel, mgrid_kernel
from .loader import Loader
from .report import Report
from .skeleton import Skeleton
from .skeleton_config import OCC, SkeletonConfig

if TYPE_CHECKING:
    import warp as wp

__all__ = [
    "Gate",
    "Index_3d",
    "System",
    "Dense",
    "DenseField",
    "Block",
    "BlockField",
    "Mgrid",
    "MgridField",
    "dense_kernel",
    "block_kernel",
    "mgrid_kernel",
    "kernel",
    "BoundKernel",
    "build_bound_kernel",
    "Loader",
    "Report",
    "Skeleton",
    "SkeletonConfig",
    "OCC",
    "init",
    "init_warp",
    "install_neon_aliases",
    "WARP_BRANCH",
    "WARP_REPO",
]

WARP_REPO = "https://github.com/nvlukasz/warp.git"
WARP_BRANCH = "external-source-support-update2"


def init_warp() -> None:
    import warp as wp

    wp.init()
    wp.build.set_cpp_standard("c++17")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    include_dir = os.path.join(repo_root, "libcarbonpy", "include")
    wp.build.add_include_directory(include_dir)
    wp.build.add_preprocessor_macro_definition("CARBON_WARP_COMPILATION")
    wp.build.clear_kernel_cache()

    from .warp_builtins import register_carbon_warp_types

    register_carbon_warp_types()


def install_neon_aliases() -> None:
    """Expose ``wp.neon_*`` builtins backed by Carbon (call after :func:`init_warp`).

    Lets code written against Neon's Warp builtins run unchanged on Carbon. Idempotent-ish:
    re-registering simply re-adds the overloads, so prefer calling once per process.
    """
    from .warp_builtins import register_neon_aliases

    register_neon_aliases()


def init(enable_warp: bool = False) -> Gate:
    gate = Gate()
    rc = gate.lib.carbonpy_init()
    if rc != 0:
        raise RuntimeError(f"carbonpy_init failed with code {rc}")
    if enable_warp:
        init_warp()
    return gate
