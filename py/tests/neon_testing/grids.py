"""Grid builders and host-side iteration helpers."""

import itertools

import neon

from . import masks

# Two-level reference grid used by several mres tests.
GRID_DIM = 16
SHELL_DEPTH = 4
LEVEL_ONE_PEEL = 2
REF_FACTOR = 2
# Fine voxels covered by one coarse cell, per axis, in this two-level layout.
COARSE_FOOTPRINT = REF_FACTOR * REF_FACTOR

ORIGIN = neon.Index_3d(0, 0, 0)
DEFAULT_STENCIL = [[0, 0, 0], [1, 0, 0]]


def coord_sum(idx) -> int:
    return idx.x + idx.y + idx.z


def iter_dim(dim):
    """Yield every ``Index_3d`` of a dense ``dim`` in z, y, x order."""
    for z, y, x in itertools.product(range(dim.z), range(dim.y), range(dim.x)):
        yield neon.Index_3d(x, y, z)


def make_mgrid(backend, dim: int, level_masks, stencil=None):
    return neon.mGrid(
        backend,
        neon.Index_3d(dim, dim, dim),
        sparsity_pattern_list=list(level_masks),
        sparsity_pattern_origins=[ORIGIN] * len(level_masks),
        stencil=stencil or DEFAULT_STENCIL,
    )


def two_level_masks():
    """Level 0: shell, level 1: peeled interior, with fine voxels under coarse cells removed."""
    fine = masks.shell_mask(GRID_DIM, SHELL_DEPTH)
    coarse = masks.peeled_mask(GRID_DIM // 2, LEVEL_ONE_PEEL)
    fine = masks.strip_fine_under_coarse(fine, coarse, COARSE_FOOTPRINT, COARSE_FOOTPRINT)
    return fine, coarse


def two_level_mgrid(backend):
    fine, coarse = two_level_masks()
    return make_mgrid(backend, GRID_DIM, [fine, coarse]), fine, coarse


def active_voxels(grid, level: int, dim: int):
    """Active voxels of ``level`` in base-resolution coordinates (stride ``2**level``)."""
    stride = 1 << level
    out = []
    for z, y, x in itertools.product(range(0, dim, stride), repeat=3):
        idx = neon.Index_3d(x, y, z)
        if grid.is_inside_domain(level, idx):
            out.append(idx)
    return out
