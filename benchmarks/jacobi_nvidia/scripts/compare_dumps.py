#!/usr/bin/env python3
"""Compare two Jacobi field dumps (row-major float32, ny * nx elements)."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


def read_dump(path: Path, nx: int, ny: int) -> list[float]:
    data = path.read_bytes()
    expected = nx * ny * 4
    if len(data) != expected:
        raise ValueError(f"{path}: expected {expected} bytes, got {len(data)}")
    return list(struct.unpack(f"<{nx * ny}f", data))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("-nx", type=int, required=True)
    parser.add_argument("-ny", type=int, required=True)
    # Cross-codebase float32 dumps: allow ~1 ULP (NVIDIA vs Carbon may differ slightly in order).
    parser.add_argument("-tol", type=float, default=1.0e-6)
    args = parser.parse_args()

    ref = read_dump(Path(args.reference), args.nx, args.ny)
    got = read_dump(Path(args.candidate), args.nx, args.ny)

    worst = 0.0
    worst_idx = 0
    for iy in range(1, args.ny - 1):
        for ix in range(1, args.nx - 1):
            off = iy * args.nx + ix
            d = abs(ref[off] - got[off])
            if d > worst:
                worst = d
                worst_idx = off
    if worst > args.tol:
        iy, ix = divmod(worst_idx, args.nx)
        print(
            f"FAIL: max abs error {worst} at iy={iy} ix={ix} "
            f"ref={ref[worst_idx]} got={got[worst_idx]}",
            file=sys.stderr,
        )
        return 1
    print(f"OK: max interior abs error {worst} <= tol {args.tol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
