#!/usr/bin/env python3
"""Unified harness for carbon micro-benchmarks (reduce, skeleton_overlap, block_sparse)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Row:
    benchmark: str
    grid: str
    overlap: str
    gpus: int
    n: int
    batch_iters: int
    metric_name: str
    metric_value: float
    elapsed_s: float
    repeats: int
    topo: str
    report_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def detect_topo() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"], text=True, stderr=subprocess.DEVNULL, timeout=10
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if len(lines) < 2:
        return "unknown"
    header = re.split(r"\s+", lines[0].strip())
    gpu_cols = [h for h in header if h.startswith("GPU")]
    if len(gpu_cols) < 2:
        return "single_gpu"
    for line in lines[1:]:
        parts = re.split(r"\s+", line.strip())
        if parts and parts[0] == "GPU0" and len(parts) >= 3:
            return parts[2]
    return "unknown"


def parse_scrape_line(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in line.strip().split():
        if "=" in token:
            key, val = token.split("=", 1)
            out[key] = val
    return out


def run_cmd(cmd: list[str], env: dict[str, str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def bench_bin(build_dir: Path, name: str) -> Path:
    return build_dir / "benchmarks" / name / f"carbon_benchmark_{name}"


def cuda_env(gpus: int, topo: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CARBON_BENCH_TOPO"] = topo
    if gpus > 0:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpus))
    return env


def append_row(rows: list[Row], **kwargs: Any) -> None:
    rows.append(Row(**kwargs))


def run_reduce(
    build_dir: Path,
    out_dir: Path,
    n: int,
    gpus: int,
    repeats: int,
    batch_iters: int,
    topo: str,
    rows: list[Row],
) -> None:
    bin_path = bench_bin(build_dir, "reduce_throughput")
    if not bin_path.is_file():
        print(f"skip reduce_throughput: missing {bin_path}", file=sys.stderr)
        return
    report_base = out_dir / f"reduce_n{n}_g{gpus}_b{batch_iters}"
    cmd = [
        str(bin_path),
        "--n",
        str(n),
        "--gpus",
        str(gpus),
        "--repeats",
        str(repeats),
        "--batch-iters",
        str(batch_iters),
        "--report",
        str(report_base),
    ]
    rc, stdout, stderr = run_cmd(cmd, cuda_env(gpus, topo))
    if rc != 0:
        print(stderr, file=sys.stderr)
        raise SystemExit(f"reduce_throughput failed (rc={rc})")
    for line in stdout.splitlines():
        if not line.startswith("REDUCE_GBs="):
            continue
        kv = parse_scrape_line(line)
        grid = kv.get("grid", "?")
        append_row(
            rows,
            benchmark="reduce_throughput",
            grid=grid,
            overlap="-",
            gpus=gpus,
            n=n,
            batch_iters=batch_iters,
            metric_name="reduce_gbs",
            metric_value=float(kv["REDUCE_GBs"]),
            elapsed_s=float(kv.get("elapsed_seconds", "0")),
            repeats=repeats,
            topo=topo,
            report_path=str(report_base) + f".{grid}",
        )


def run_skeleton_overlap(
    build_dir: Path,
    out_dir: Path,
    n: int,
    gpus: int,
    repeats: int,
    topo: str,
    rows: list[Row],
) -> None:
    bin_path = bench_bin(build_dir, "skeleton_overlap")
    if not bin_path.is_file():
        print(f"skip skeleton_overlap: missing {bin_path}", file=sys.stderr)
        return
    report_base = out_dir / f"skeleton_n{n}_g{gpus}"
    cmd = [
        str(bin_path),
        "--n",
        str(n),
        "--gpus",
        str(gpus),
        "--repeats",
        str(repeats),
        "--report",
        str(report_base),
    ]
    rc, stdout, stderr = run_cmd(cmd, cuda_env(gpus, topo))
    if stderr:
        print(stderr, file=sys.stderr)
    if rc != 0:
        raise SystemExit(f"skeleton_overlap failed (rc={rc})")
    for line in stdout.splitlines():
        if not line.startswith("MLUPS="):
            continue
        kv = parse_scrape_line(line)
        overlap = kv.get("overlap", "?")
        append_row(
            rows,
            benchmark="skeleton_overlap",
            grid="dense",
            overlap=overlap,
            gpus=gpus,
            n=n,
            batch_iters=1,
            metric_name="mlups",
            metric_value=float(kv["MLUPS"]),
            elapsed_s=float(kv.get("elapsed_seconds", "0")),
            repeats=repeats,
            topo=topo,
            report_path=str(report_base) + f".{overlap}",
        )


def run_block_sparse(
    build_dir: Path,
    out_dir: Path,
    n: int,
    gpus: int,
    repeats: int,
    topo: str,
    rows: list[Row],
) -> None:
    bin_path = bench_bin(build_dir, "block_sparse")
    if not bin_path.is_file():
        print(f"skip block_sparse: missing {bin_path}", file=sys.stderr)
        return
    report_path = out_dir / f"block_sparse_n{n}_g{gpus}.json"
    cmd = [
        str(bin_path),
        "--n",
        str(n),
        "--gpus",
        str(gpus),
        "--repeats",
        str(repeats),
        "--report",
        str(report_path),
    ]
    rc, stdout, stderr = run_cmd(cmd, cuda_env(gpus, topo))
    if rc != 0:
        print(stderr, file=sys.stderr)
        raise SystemExit(f"block_sparse failed (rc={rc})")
    for line in stdout.splitlines():
        if not line.startswith("MLUPS="):
            continue
        kv = parse_scrape_line(line)
        append_row(
            rows,
            benchmark="block_sparse",
            grid="block",
            overlap="-",
            gpus=gpus,
            n=n,
            batch_iters=1,
            metric_name="mlups",
            metric_value=float(kv["MLUPS"]),
            elapsed_s=float(kv.get("elapsed_seconds", "0")),
            repeats=repeats,
            topo=topo,
            report_path=str(report_path),
            extra={
                "halo_pair_count": int(kv.get("halo_pair_count", "0")),
                "active_fraction": float(kv.get("active_fraction", "0")),
            },
        )


def write_summary(rows: list[Row], out_dir: Path) -> None:
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "benchmark",
                "grid",
                "overlap",
                "gpus",
                "n",
                "batch_iters",
                "metric_name",
                "metric_value",
                "elapsed_s",
                "repeats",
                "topo",
                "report_path",
                "extra_json",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.benchmark,
                    r.grid,
                    r.overlap,
                    r.gpus,
                    r.n,
                    r.batch_iters,
                    r.metric_name,
                    f"{r.metric_value:.6g}",
                    f"{r.elapsed_s:.6g}",
                    r.repeats,
                    r.topo,
                    r.report_path,
                    json.dumps(r.extra),
                ]
            )

    md_path = out_dir / "summary.md"
    with md_path.open("w") as f:
        f.write("# Carbon benchmark summary\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
        f.write("| benchmark | grid | overlap | gpus | n | batch | metric | value | elapsed_s | topo |\n")
        f.write("|-----------|------|---------|------|---|-------|--------|-------|-----------|------|\n")
        for r in rows:
            batch = r.batch_iters if r.benchmark == "reduce_throughput" else "-"
            f.write(
                f"| {r.benchmark} | {r.grid} | {r.overlap} | {r.gpus} | {r.n} | {batch} | "
                f"{r.metric_name} | {r.metric_value:.4g} | {r.elapsed_s:.4g} | {r.topo} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run carbon micro-benchmark matrix")
    parser.add_argument("--build-dir", default="build", type=Path)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Report output directory (default reports/bench-YYYY-MM-DD)",
    )
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    args = parser.parse_args()

    build_dir: Path = args.build_dir
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("reports") / f"bench-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    topo = detect_topo()
    print(f"interconnect_topo={topo}")

    if args.mode == "quick":
        sizes = [256]
        gpus_list = [1, 2]
        repeats = 3
    else:
        sizes = [128, 192, 256]
        gpus_list = [1, 2]
        repeats = 5

    rows: list[Row] = []
    for n in sizes:
        for gpus in gpus_list:
            print(f">>> reduce_throughput n={n} gpus={gpus} batch=1")
            run_reduce(build_dir, out_dir, n, gpus, repeats, 1, topo, rows)
            print(f">>> reduce_throughput n={n} gpus={gpus} batch=32")
            run_reduce(build_dir, out_dir, n, gpus, repeats, 32, topo, rows)
            print(f">>> skeleton_overlap n={n} gpus={gpus}")
            run_skeleton_overlap(build_dir, out_dir, n, gpus, repeats, topo, rows)
            print(f">>> block_sparse n={n} gpus={gpus}")
            run_block_sparse(build_dir, out_dir, n, gpus, repeats, topo, rows)

    write_summary(rows, out_dir)
    print(f"wrote {out_dir / 'summary.csv'} and {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
