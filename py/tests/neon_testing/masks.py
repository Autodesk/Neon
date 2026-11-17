"""Sparsity masks for mGrid tests. Masks are indexed ``mask[x, y, z]`` (int32, C order)."""

import numpy as np


def _as_mask(boolean: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(boolean, dtype=np.int32)


def _axes(shape):
    return np.indices(shape, dtype=np.float64)


def shell_mask(dim: int, depth: int) -> np.ndarray:
    """Cube of edge ``dim`` with the outer ``depth`` layers active."""
    x, y, z = _axes((dim, dim, dim))
    lo, hi = depth, dim - depth
    outside = lambda a: (a < lo) | (a >= hi)  # noqa: E731
    return _as_mask(outside(x) | outside(y) | outside(z))


def peeled_mask(dim: int, depth: int) -> np.ndarray:
    """Cube of edge ``dim`` with everything active except the outer ``depth`` layers."""
    return _as_mask(1 - shell_mask(dim, depth))


def outer_shell(shape, peel) -> np.ndarray:
    """Active where any coordinate is within ``peel`` of a face (``peel`` may be fractional)."""
    x, y, z = _axes(shape)
    nx, ny, nz = shape
    near = lambda a, n: (a <= peel) | (a >= n - 1 - peel)  # noqa: E731
    return _as_mask(near(x, nx) | near(y, ny) | near(z, nz))


def inner_box(shape, peel) -> np.ndarray:
    """Active where every coordinate is at least ``peel`` away from both faces."""
    x, y, z = _axes(shape)
    nx, ny, nz = shape
    inside = lambda a, n: (a >= peel) & (a <= n - 1 - peel)  # noqa: E731
    return _as_mask(inside(x, nx) & inside(y, ny) & inside(z, nz))


def strip_fine_under_coarse(fine: np.ndarray, coarse: np.ndarray, block_spacing: int, child_span: int) -> np.ndarray:
    """Clear fine voxels covered by active coarse blocks so the levels do not overlap."""
    fine = fine.copy()
    for bx, by, bz in np.argwhere(coarse == 1):
        ox, oy, oz = bx * block_spacing, by * block_spacing, bz * block_spacing
        fine[ox:ox + child_span, oy:oy + child_span, oz:oz + child_span] = 0
    return _as_mask(fine)


def nested_shell_levels(dim: int, widths) -> list:
    """Masks for a ``len(widths) + 1`` level grid: nested shells, full coarsest level.

    Level ``l`` has edge ``dim // 2**l`` and a shell of thickness ``edge / widths[l]``.
    The coarsest level has edge ``dim // 2**(levels - 1) + 1`` and is fully active.
    """
    levels = []
    for level, width in enumerate(widths):
        edge = dim // (2 ** level)
        levels.append(outer_shell((edge, edge, edge), edge / width))
    coarse = dim // (2 ** len(widths)) + 1
    levels.append(_as_mask(np.ones((coarse, coarse, coarse))))
    return levels
