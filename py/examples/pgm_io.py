"""Minimal PGM (P5) reader/writer for Carbon Python examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class GrayImage:
    width: int
    height: int
    pixels: list[float]


def _skip_comments_and_ws(data: bytes, pos: int) -> int:
    while pos < len(data):
        if data[pos : pos + 1].isspace():
            pos += 1
            continue
        if data[pos : pos + 1] == b"#"[0:1]:
            while pos < len(data) and data[pos : pos + 1] != b"\n"[0:1]:
                pos += 1
            continue
        break
    return pos


def _read_int(data: bytes, pos: int) -> tuple[int, int]:
    pos = _skip_comments_and_ws(data, pos)
    end = pos
    while end < len(data) and data[end : end + 1].isdigit():
        end += 1
    if end == pos:
        raise ValueError("expected integer in PGM header")
    return int(data[pos:end].decode("ascii")), end


def load_pgm(path: str | Path) -> GrayImage:
    path = Path(path)
    data = path.read_bytes()
    if not data.startswith(b"P5"):
        raise ValueError(f"{path}: expected binary PGM (P5)")

    pos = 2
    width, pos = _read_int(data, pos)
    height, pos = _read_int(data, pos)
    maxval, pos = _read_int(data, pos)
    if width <= 0 or height <= 0 or maxval <= 0:
        raise ValueError(f"{path}: malformed PGM header")

    while pos < len(data) and data[pos : pos + 1].isspace():
        pos += 1

    inv = 1.0 / float(maxval)
    count = width * height
    if maxval < 256:
        raw = data[pos : pos + count]
        if len(raw) != count:
            raise ValueError(f"{path}: short read on 8-bit pixel data")
        pixels = [float(b) * inv for b in raw]
    else:
        raw = data[pos : pos + count * 2]
        if len(raw) != count * 2:
            raise ValueError(f"{path}: short read on 16-bit pixel data")
        pixels = []
        for i in range(count):
            hi = raw[2 * i]
            lo = raw[2 * i + 1]
            pixels.append(float((hi << 8) | lo) * inv)
    return GrayImage(width=width, height=height, pixels=pixels)


def save_pgm(path: str | Path, width: int, height: int, pixels: list[float]) -> None:
    path = Path(path)
    if len(pixels) != width * height:
        raise ValueError("pixel count does not match width * height")
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    body = bytearray(len(pixels))
    for i, v in enumerate(pixels):
        body[i] = max(0, min(255, int(round(max(0.0, min(1.0, v)) * 255.0))))
    path.write_bytes(header + bytes(body))
