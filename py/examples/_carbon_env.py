"""Shared bootstrap for Carbon Python examples."""

from __future__ import annotations

import os
import sys


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def setup_paths() -> str:
    root = repo_root()
    py_dir = os.path.join(root, "py")
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)
    return root


def require_warp():
    try:
        import warp as wp  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Warp is not installed. Run ./scripts/setup_warp.sh from the repo root."
        ) from exc

    import warp as wp

    repo = getattr(wp, "__file__", "") or ""
    if "nvlukasz" not in repo and "carbon/warp" not in repo.replace("\\", "/"):
        raise SystemExit(
            "Carbon requires the external-type Warp fork. Run ./scripts/setup_warp.sh"
        )
    return wp


def require_libcarbonpy(root: str) -> None:
    import platform

    lib_path = os.path.join(
        root,
        "py",
        "carbon",
        "lib",
        f"linux_{platform.machine()}",
        "libcarbonpy.so",
    )
    if not os.path.isfile(lib_path):
        raise SystemExit(
            f"libcarbonpy not found at {lib_path}\n"
            "Build with: cmake -S . -B build-py -DCARBON_ENABLE_PYTHON=ON && cmake --build build-py -j"
        )
