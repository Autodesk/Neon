#!/usr/bin/env python3
"""Python port of examples/salt_pepper_filter.cu (Dense grid, single GPU).

Loads a grayscale PGM, corrupts it with salt-and-pepper noise, applies a 3×3
median filter in the x/z plane via Carbon partition builtins, and
reports PSNR before/after denoising. Optional PGM output mirrors the C++ CLI.

Layout matches the C++ example: the image is stored as a Dense grid with extent
(width, 1, height) so multi-GPU runs would split along image rows (z).
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np

from _carbon_env import require_libcarbonpy, require_warp, repo_root, setup_paths
from pgm_io import load_pgm, save_pgm

setup_paths()
from carbon.warp_types import CarbonDensePartition, CarbonDenseSpan  # noqa: E402


@dataclass
class CliOptions:
    input_path: str
    output_path: str | None
    noisy_output_path: str | None
    salt_prob: float
    pepper_prob: float
    seed: int


class Lcg:
    def __init__(self, seed: int) -> None:
        self._state = seed & 0xFFFFFFFF

    def next_unit(self) -> float:
        self._state = (self._state * 1664525 + 1013904223) & 0xFFFFFFFF
        return float(self._state) * (1.0 / 4294967296.0)


def parse_args(argv: list[str] | None = None) -> CliOptions:
    root = repo_root()
    default_input = os.path.join(root, "examples", "data", "astronaut.pgm")
    parser = argparse.ArgumentParser(
        description="Carbon Python salt-and-pepper median filter (Dense grid)"
    )
    parser.add_argument(
        "--input",
        default=default_input,
        help=f"PGM (P5) input image (default: {default_input})",
    )
    parser.add_argument("--output", default=None, help="Write filtered PGM (single GPU only)")
    parser.add_argument(
        "--noisy-output", default=None, help="Write noisy PGM (single GPU only)"
    )
    parser.add_argument("--salt-prob", type=float, default=0.05, help="White pixel fraction")
    parser.add_argument(
        "--pepper-prob", type=float, default=0.05, help="Black pixel fraction"
    )
    parser.add_argument(
        "--seed", type=lambda s: int(s, 0), default=0xC0FFEE, help="LCG seed (default 0xC0FFEE)"
    )
    args = parser.parse_args(argv)
    if args.salt_prob + args.pepper_prob > 1.0:
        parser.error("--salt-prob + --pepper-prob exceeds 1.0")
    return CliOptions(
        input_path=args.input,
        output_path=args.output,
        noisy_output_path=args.noisy_output,
        salt_prob=args.salt_prob,
        pepper_prob=args.pepper_prob,
        seed=args.seed,
    )


def apply_noise(pixels: list[float], opts: CliOptions) -> tuple[list[float], int, int]:
    rng = Lcg(opts.seed)
    pepper_thr = opts.pepper_prob
    salt_thr = opts.pepper_prob + opts.salt_prob
    noisy = list(pixels)
    pepper_count = 0
    salt_count = 0
    for i in range(len(noisy)):
        r = rng.next_unit()
        if r < pepper_thr:
            noisy[i] = 0.0
            pepper_count += 1
        elif r < salt_thr:
            noisy[i] = 1.0
            salt_count += 1
    return noisy, pepper_count, salt_count


def psnr_db(mse: float) -> float:
    if mse <= 0.0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def main(argv: list[str] | None = None) -> int:
    opts = parse_args(argv)
    root = setup_paths()
    require_libcarbonpy(root)
    wp = require_warp()

    import carbon  # noqa: WPS433

    img = load_pgm(opts.input_path)
    noisy_pixels, pepper_count, salt_count = apply_noise(img.pixels, opts)
    width, height = img.width, img.height

    gate = carbon.init(enable_warp=True)
    n_gpus = gate.lib.carbonpy_device_count()
    if n_gpus <= 0:
        print("salt_pepper_filter: no CUDA devices visible")
        return 1
    if n_gpus > 1:
        print(
            f"salt_pepper_filter: using GPU 0 of {n_gpus} "
            "(PGM output requires a single visible GPU)"
        )

    system = carbon.System(gate)
    dense = carbon.Dense(system, width, 1, height)
    clean_field = dense.new_field(1, dtype=wp.float32)
    noisy_field = dense.new_field(1, dtype=wp.float32)
    out_field = dense.new_field(1, dtype=wp.float32)

    clean_span = dense.get_span()
    noisy_span = dense.get_span()
    out_span = dense.get_span()
    clean_part = clean_field.get_partition()
    noisy_part = noisy_field.get_partition()
    out_part = out_field.get_partition()

    nx = int(clean_span.iter_x)
    ny = int(clean_span.iter_y)
    nz = int(clean_span.iter_z)

    clean_wp = wp.array(np.asarray(img.pixels, dtype=np.float32), device="cuda:0")
    noisy_wp = wp.array(np.asarray(noisy_pixels, dtype=np.float32), device="cuda:0")

    @wp.kernel
    def upload_row_image(
        src: wp.array(dtype=wp.float32),
        part: CarbonDensePartition,
        span: CarbonDenseSpan,
        img_width: int,
    ):
        ix, iy, iz = wp.tid()
        pixel = src[iz * img_width + ix]
        cell = wp.carbon.set(span, ix, iy, iz)
        wp.carbon.write(part, span, cell, 0, pixel)

    wp.launch(
        upload_row_image,
        dim=(nx, ny, nz),
        inputs=[clean_wp, clean_part, clean_span, width],
        device="cuda:0",
    )
    wp.launch(
        upload_row_image,
        dim=(nx, ny, nz),
        inputs=[noisy_wp, noisy_part, noisy_span, width],
        device="cuda:0",
    )
    wp.synchronize()

    @wp.func
    def read_neighbour(
        part: CarbonDensePartition,
        span: CarbonDenseSpan,
        ix: int,
        iy: int,
        iz: int,
        dx: int,
        dz: int,
        img_width: int,
        img_height: int,
    ) -> float:
        x = ix + dx
        z = iz + dz
        if x < 0 or x >= img_width or z < 0 or z >= img_height:
            return wp.float32(-1.0)
        cell = wp.carbon.set(span, x, iy, z)
        return wp.carbon.read(part, span, cell, 0)

    @wp.kernel
    def median_filter(
        in_part: CarbonDensePartition,
        in_span: CarbonDenseSpan,
        out_part: CarbonDensePartition,
        out_span: CarbonDenseSpan,
        img_width: int,
        img_height: int,
    ):
        ix, iy, iz = wp.tid()
        cell = wp.carbon.set(in_span, ix, iy, iz)
        b0 = wp.carbon.read(in_part, in_span, cell, 0)
        n = int(1)
        b1 = b0
        b2 = b0
        b3 = b0
        b4 = b0
        b5 = b0
        b6 = b0
        b7 = b0
        b8 = b0

        v = read_neighbour(in_part, in_span, ix, iy, iz, 1, 0, img_width, img_height)
        if v >= wp.float32(0.0):
            if n == 1:
                b1 = v
            elif n == 2:
                b2 = v
            elif n == 3:
                b3 = v
            elif n == 4:
                b4 = v
            elif n == 5:
                b5 = v
            elif n == 6:
                b6 = v
            elif n == 7:
                b7 = v
            elif n == 8:
                b8 = v
            n = n + 1
        v = read_neighbour(in_part, in_span, ix, iy, iz, -1, 0, img_width, img_height)
        if v >= wp.float32(0.0):
            if n == 1:
                b1 = v
            elif n == 2:
                b2 = v
            elif n == 3:
                b3 = v
            elif n == 4:
                b4 = v
            elif n == 5:
                b5 = v
            elif n == 6:
                b6 = v
            elif n == 7:
                b7 = v
            elif n == 8:
                b8 = v
            n = n + 1
        v = read_neighbour(in_part, in_span, ix, iy, iz, 0, 1, img_width, img_height)
        if v >= wp.float32(0.0):
            if n == 1:
                b1 = v
            elif n == 2:
                b2 = v
            elif n == 3:
                b3 = v
            elif n == 4:
                b4 = v
            elif n == 5:
                b5 = v
            elif n == 6:
                b6 = v
            elif n == 7:
                b7 = v
            elif n == 8:
                b8 = v
            n = n + 1
        v = read_neighbour(in_part, in_span, ix, iy, iz, 0, -1, img_width, img_height)
        if v >= wp.float32(0.0):
            if n == 1:
                b1 = v
            elif n == 2:
                b2 = v
            elif n == 3:
                b3 = v
            elif n == 4:
                b4 = v
            elif n == 5:
                b5 = v
            elif n == 6:
                b6 = v
            elif n == 7:
                b7 = v
            elif n == 8:
                b8 = v
            n = n + 1
        v = read_neighbour(in_part, in_span, ix, iy, iz, 1, 1, img_width, img_height)
        if v >= wp.float32(0.0):
            if n == 1:
                b1 = v
            elif n == 2:
                b2 = v
            elif n == 3:
                b3 = v
            elif n == 4:
                b4 = v
            elif n == 5:
                b5 = v
            elif n == 6:
                b6 = v
            elif n == 7:
                b7 = v
            elif n == 8:
                b8 = v
            n = n + 1
        v = read_neighbour(in_part, in_span, ix, iy, iz, 1, -1, img_width, img_height)
        if v >= wp.float32(0.0):
            if n == 1:
                b1 = v
            elif n == 2:
                b2 = v
            elif n == 3:
                b3 = v
            elif n == 4:
                b4 = v
            elif n == 5:
                b5 = v
            elif n == 6:
                b6 = v
            elif n == 7:
                b7 = v
            elif n == 8:
                b8 = v
            n = n + 1
        v = read_neighbour(in_part, in_span, ix, iy, iz, -1, 1, img_width, img_height)
        if v >= wp.float32(0.0):
            if n == 1:
                b1 = v
            elif n == 2:
                b2 = v
            elif n == 3:
                b3 = v
            elif n == 4:
                b4 = v
            elif n == 5:
                b5 = v
            elif n == 6:
                b6 = v
            elif n == 7:
                b7 = v
            elif n == 8:
                b8 = v
            n = n + 1
        v = read_neighbour(in_part, in_span, ix, iy, iz, -1, -1, img_width, img_height)
        if v >= wp.float32(0.0):
            if n == 1:
                b1 = v
            elif n == 2:
                b2 = v
            elif n == 3:
                b3 = v
            elif n == 4:
                b4 = v
            elif n == 5:
                b5 = v
            elif n == 6:
                b6 = v
            elif n == 7:
                b7 = v
            elif n == 8:
                b8 = v
            n = n + 1

        if n > 1 and b1 < b0:
            t = b0
            b0 = b1
            b1 = t
        if n > 2 and b2 < b1:
            t = b1
            b1 = b2
            b2 = t
            if b1 < b0:
                t = b0
                b0 = b1
                b1 = t
        if n > 3 and b3 < b2:
            t = b2
            b2 = b3
            b3 = t
            if b2 < b1:
                t = b1
                b1 = b2
                b2 = t
                if b1 < b0:
                    t = b0
                    b0 = b1
                    b1 = t
        if n > 4 and b4 < b3:
            t = b3
            b3 = b4
            b4 = t
            if b3 < b2:
                t = b2
                b2 = b3
                b3 = t
                if b2 < b1:
                    t = b1
                    b1 = b2
                    b2 = t
                    if b1 < b0:
                        t = b0
                        b0 = b1
                        b1 = t
        if n > 5 and b5 < b4:
            t = b4
            b4 = b5
            b5 = t
        if n > 6 and b6 < b5:
            t = b5
            b5 = b6
            b6 = t
        if n > 7 and b7 < b6:
            t = b6
            b6 = b7
            b7 = t
        if n > 8 and b8 < b7:
            t = b7
            b7 = b8
            b8 = t

        mid = n // 2
        median = b0
        if mid == 1:
            median = b1
        if mid == 2:
            median = b2
        if mid == 3:
            median = b3
        if mid == 4:
            median = b4
        out_cell = wp.carbon.set(out_span, ix, iy, iz)
        wp.carbon.write(out_part, out_span, out_cell, 0, median)

    wp.launch(
        median_filter,
        dim=(nx, ny, nz),
        inputs=[noisy_part, noisy_span, out_part, out_span, width, height],
        device="cuda:0",
    )
    wp.synchronize()

    @wp.kernel
    def copy_field(
        part: CarbonDensePartition,
        span: CarbonDenseSpan,
        img_width: int,
        out: wp.array(dtype=wp.float32),
    ):
        ix, iy, iz = wp.tid()
        cell = wp.carbon.set(span, ix, iy, iz)
        out[iz * img_width + ix] = wp.carbon.read(part, span, cell, 0)

    flat_count = width * height
    clean_gpu = wp.zeros(flat_count, dtype=wp.float32, device="cuda:0")
    noisy_gpu = wp.zeros(flat_count, dtype=wp.float32, device="cuda:0")
    filt_gpu = wp.zeros(flat_count, dtype=wp.float32, device="cuda:0")

    wp.launch(
        copy_field,
        dim=(nx, ny, nz),
        inputs=[clean_part, clean_span, width, clean_gpu],
        device="cuda:0",
    )
    wp.launch(
        copy_field,
        dim=(nx, ny, nz),
        inputs=[noisy_part, noisy_span, width, noisy_gpu],
        device="cuda:0",
    )
    wp.launch(
        copy_field,
        dim=(nx, ny, nz),
        inputs=[out_part, out_span, width, filt_gpu],
        device="cuda:0",
    )
    wp.synchronize()

    clean_np = clean_gpu.numpy()
    noisy_np = noisy_gpu.numpy()
    filt_np = filt_gpu.numpy()

    mse_noisy = float(np.mean((noisy_np - clean_np) ** 2))
    mse_filtered = float(np.mean((filt_np - clean_np) ** 2))

    print(f"salt_pepper_filter: loaded {opts.input_path} ({width} x {height})")
    print(
        f"salt_pepper_filter: seed=0x{opts.seed:08x} pepper={pepper_count} salt={salt_count}"
    )
    print(
        f"salt_pepper_filter: PSNR noisy    vs clean = {psnr_db(mse_noisy):6.2f} dB "
        f"(MSE = {mse_noisy:.4e})"
    )
    print(
        f"salt_pepper_filter: PSNR filtered vs clean = {psnr_db(mse_filtered):6.2f} dB "
        f"(MSE = {mse_filtered:.4e})"
    )

    ok = mse_filtered < mse_noisy
    if not ok:
        print(
            "salt_pepper_filter: filter did not reduce MSE "
            f"({mse_filtered:.4e} >= {mse_noisy:.4e})"
        )

    if n_gpus == 1:
        if opts.noisy_output_path:
            save_pgm(opts.noisy_output_path, width, height, noisy_pixels)
            print(f"salt_pepper_filter: wrote noisy image to {opts.noisy_output_path}")
        if opts.output_path:
            save_pgm(opts.output_path, width, height, filt_np.tolist())
            print(f"salt_pepper_filter: wrote filtered image to {opts.output_path}")

    clean_field.delete()
    noisy_field.delete()
    out_field.delete()
    dense.delete()
    system.delete()

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
