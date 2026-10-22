#!/usr/bin/env python3
"""Sweep the lbm_lid_driven_cavity benchmark and plot the results.

Two related operations, both driven by the JSON report files emitted by the
benchmark binary's ``--report PATH`` flag:

  * ``sweep`` — run the benchmark across a grid of (N, gpus, lattice,
    collision) and write one ``<output_dir>/run-<index>.json`` per run.
  * ``plot``  — load every ``*.json`` under one or more report directories
    and render a Neon-style figure: MLUPS on the y-axis vs N (size sweep)
    or vs the GPU count (strong scaling), with one curve per
    (lattice, collision, label, gpu_count) series.

Both modes can run together (``sweep`` then ``plot``); ``plot`` standalone is
useful to re-render an existing report directory or to compare two sweeps
side by side. The script depends only on the standard library plus
``matplotlib`` (and ``numpy``, transitively).

Schema versioning: the benchmark binary writes ``schema:
"carbon-lbm-report-v1"``. If you bump that string in ``lbm_lid_driven_cavity.cu``,
update :data:`SCHEMA` here in lockstep so old reports do not silently mix
with new ones.

Example — single-GPU size sweep then plot:

    python3 benchmarks/lbm_lid_driven_cavity/scripts/sweep_and_plot.py \\
        --bench ./build/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity \\
        --mode sweep+plot \\
        --output-dir reports/single_gpu \\
        --sizes 64 96 128 160 192 224 256 288 320 384 \\
        --gpus 1 \\
        --lattice q19 --collision bgk \\
        --iters 600 --warmup 200 \\
        --plot-x size \\
        --plot-out reports/single_gpu/mlups_vs_n.png

Example — strong-scaling sweep at N=256 across 1, 2, ..., G GPUs:

    python3 benchmarks/lbm_lid_driven_cavity/scripts/sweep_and_plot.py \\
        --bench ./build/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity \\
        --mode sweep+plot \\
        --output-dir reports/scaling_n256 \\
        --sizes 256 \\
        --gpus 1 2 \\
        --lattice q19 --collision bgk \\
        --plot-x gpus --plot-out reports/scaling_n256/strong.png

Example — single-GPU N=256 across every valid (lattice, collision) config,
rendered as a bar chart:

    python3 benchmarks/lbm_lid_driven_cavity/scripts/sweep_and_plot.py \\
        --bench ./build/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity \\
        --mode sweep+plot \\
        --output-dir reports/single_gpu_n256 \\
        --sizes 256 \\
        --gpus 1 \\
        --configs q19:bgk q27:bgk q27:kbc \\
        --plot-x config --plot-out reports/single_gpu_n256/mlups_by_config.png
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "carbon-lbm-report-v1"


# All inputs to this script come from a developer-side argparse / CLI; we
# never accept JSON content from an untrusted network source. Reports are
# produced by the C++ binary in this same repo. The only "external" input is
# the benchmark binary's own stdout, which we don't shell-pipe — we exec it
# directly via subprocess with ``shell=False``.


@dataclass(frozen=True)
class RunSpec:
    """Single benchmark invocation."""

    n: int
    gpus: int
    lattice: str
    collision: str
    iters: int
    warmup: int
    re: float
    ulb: float | None
    label: str
    # Logical name of the backend this RunSpec is dispatched to. Used purely
    # as a tag — the actual binary path is looked up from the parsed
    # --backends mapping at sweep time. Empty string means "use --bench"
    # (single-backend backwards-compatible mode).
    backend: str = ""
    overlap: str = "none"


@dataclass
class Report:
    """Parsed contents of one report JSON. Only the fields plot needs."""

    n: int
    gpus: int
    lattice: str
    collision: str
    label: str
    backend: str
    overlap: str
    mlups: float
    mlups_per_gpu: float
    timing_seconds: float
    timed_iters: int
    effective_bandwidth_gb_s: float
    gpu_name: str
    raw: dict[str, Any] = field(repr=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep + plot the carbon LBM lid-driven cavity benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--mode",
        choices=("sweep", "plot", "sweep+plot"),
        default="sweep+plot",
        help="What to do (default: sweep then plot).",
    )
    p.add_argument(
        "--bench",
        type=Path,
        default=Path(
            "build/benchmarks/lbm_lid_driven_cavity/"
            "carbon_benchmark_lbm_lid_driven_cavity"
        ),
        help="Path to the carbon_benchmark_lbm_lid_driven_cavity binary. "
        "Used when --backends is not set (single-backend sweep). The report "
        "JSON's `system.backend` field still self-identifies as 'cuda' / "
        "'sycl' based on how the binary was compiled.",
    )
    p.add_argument(
        "--backends",
        nargs="+",
        default=None,
        metavar="NAME=PATH",
        help="Sweep across multiple backend binaries, each tagged by NAME "
        "(typically 'cuda' / 'sycl'). Example: "
        "--backends cuda=build/benchmarks/.../carbon_benchmark_lbm_lid_driven_cavity "
        "sycl=build-sycl/benchmarks/.../carbon_benchmark_lbm_lid_driven_cavity . "
        "Each (backend, lattice, collision, gpus, n) tuple becomes one run; "
        "report filenames are prefixed with the backend so reports/ stays "
        "unambiguous when both backends write into the same output dir. "
        "Overrides --bench when set.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write per-run JSON reports into (sweep mode) "
        "and/or to read them from (plot mode).",
    )
    p.add_argument(
        "--input-dirs",
        type=Path,
        nargs="*",
        default=None,
        help="Plot mode only: directories to scan for *.json reports. "
        "Defaults to [--output-dir] when omitted.",
    )
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[64, 96, 128, 160, 192, 224, 256],
        help="Cavity edge length(s) N to sweep. Default: a balanced size "
        "ladder up to 256.",
    )
    p.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[1],
        help="GPU counts to sweep (each requires that many devices on this "
        "node; selected via CUDA_VISIBLE_DEVICES=0,1,...,g-1).",
    )
    p.add_argument(
        "--lattice",
        choices=("q19", "q27"),
        default="q19",
        help="Lattice stencil (passed verbatim to --lattice). Ignored when "
        "--configs is supplied.",
    )
    p.add_argument(
        "--collision",
        choices=("bgk", "kbc"),
        default="bgk",
        help="Collision model (passed verbatim to --collision). Ignored when "
        "--configs is supplied.",
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=None,
        metavar="LATTICE:COLLISION",
        help="Sweep multiple (lattice, collision) pairs in one invocation, "
        "e.g. --configs q19:bgk q27:bgk q27:kbc . When set, overrides the "
        "single --lattice/--collision pair. Useful with --plot-x config to "
        "compare configs side by side at a fixed (N, gpu_count).",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=600,
        help="Total iterations per run (default 600).",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=200,
        help="Warmup iterations excluded from timing (default 200).",
    )
    p.add_argument(
        "--re",
        type=float,
        default=100.0,
        help="Reynolds number (default 100).",
    )
    p.add_argument(
        "--ulb",
        type=float,
        default=None,
        help="Lid velocity in lattice units. Defaults to the binary default.",
    )
    p.add_argument(
        "--label",
        type=str,
        default="",
        help="Free-form label tagged into config.label of every report.",
    )
    p.add_argument(
        "--overlaps",
        nargs="+",
        default=["none"],
        choices=("none", "standard", "extended", "twoway-extended"),
        help="Skeleton OverlapMode values to sweep (default: none). Passed "
        "to the binary as --overlap.",
    )
    p.add_argument(
        "--plot-x",
        choices=("size", "gpus", "config"),
        default="size",
        help="X-axis: 'size' (MLUPS vs N, one line per gpu count), 'gpus' "
        "(MLUPS vs gpu count, one line per N), or 'config' (bar chart, one "
        "bar per (lattice, collision) pair — natural pairing with --configs "
        "at a fixed (N, gpu_count)).",
    )
    p.add_argument(
        "--plot-y",
        choices=("mlups", "mlups_per_gpu", "bandwidth"),
        default="mlups",
        help="Y-axis: 'mlups' (default), 'mlups_per_gpu' (good for scaling "
        "efficiency), or 'bandwidth' (effective HBM GB/s).",
    )
    p.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="Save the plot to this PNG/PDF/SVG. If omitted, the figure is "
        "written next to the input dir as 'mlups_<x>.png'.",
    )
    p.add_argument(
        "--plot-title",
        type=str,
        default=None,
        help="Override the auto-generated plot title.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Pop the figure interactively (in addition to saving it).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print every benchmark invocation but do not execute (sweep mode).",
    )
    return p.parse_args()


def run_one(bench: Path, spec: RunSpec, report_path: Path, dry_run: bool) -> None:
    """Invoke the benchmark binary for one ``RunSpec``."""
    if not bench.is_file():
        raise FileNotFoundError(f"benchmark binary not found: {bench}")
    env = os.environ.copy()
    # CUDA_VISIBLE_DEVICES restricts the device set the binary sees. With
    # AdaptiveCpp + cuda target this also covers the SYCL path because the
    # AdaptiveCpp CUDA backend honours the same envvar; if you ever run a
    # SYCL build on AMD / oneAPI, swap in HIP_VISIBLE_DEVICES /
    # ONEAPI_DEVICE_SELECTOR — both of those are picked up by their
    # respective runtimes alongside CUDA_VISIBLE_DEVICES, so leaving it
    # set on a non-CUDA SYCL run is harmless.
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(spec.gpus))
    cmd: list[str] = [
        str(bench),
        "--n", str(spec.n),
        "--iters", str(spec.iters),
        "--warmup", str(spec.warmup),
        "--re", str(spec.re),
        "--lattice", spec.lattice,
        "--collision", spec.collision,
        "--overlap", spec.overlap,
        "--report-every", "0",
        "--report", str(report_path),
    ]
    # `--label` is optional; clipp's value parser rejects empty strings, so
    # only forward the flag when the caller supplied a non-empty label.
    if spec.label:
        cmd.extend(["--label", spec.label])
    if spec.ulb is not None:
        cmd.extend(["--ulb", str(spec.ulb)])
    backend_tag = f"backend={spec.backend} " if spec.backend else ""
    pretty = (
        f"  {backend_tag}N={spec.n:<4d} gpus={spec.gpus} "
        f"lattice={spec.lattice} collision={spec.collision} "
        f"overlap={spec.overlap} "
        f"-> {report_path.name}"
    )
    print(pretty, flush=True)
    if dry_run:
        print("    dry-run cmd:", " ".join(cmd))
        return
    res = subprocess.run(  # noqa: S603 — bench is a developer-supplied path; shell=False
        cmd, env=env, check=False, capture_output=True, text=True
    )
    if res.returncode != 0:
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        raise RuntimeError(
            f"benchmark exited with code {res.returncode} for spec {spec}"
        )


def _resolve_backends(args: argparse.Namespace) -> list[tuple[str, Path]]:
    """Normalise --backends / --bench into a list of (name, binary_path).

    --backends NAME=PATH ... wins when set. NAME is a free-form tag (we
    recommend 'cuda' / 'sycl' so it lines up with the binary's
    `system.backend` self-tag). Returns at least one entry; raises
    SystemExit on a malformed token. When --backends is empty / unset,
    falls back to a single ('', --bench) tuple — the empty backend tag
    means "let the report's `system.backend` field do the talking".
    """
    if args.backends:
        out: list[tuple[str, Path]] = []
        seen: set[str] = set()
        for tok in args.backends:
            if "=" not in tok:
                raise SystemExit(
                    f"--backends token '{tok}' must be NAME=PATH "
                    f"(e.g. cuda=build/.../carbon_benchmark_lbm_lid_driven_cavity)"
                )
            name, _, path = tok.partition("=")
            name = name.strip().lower()
            path = path.strip()
            if not name or not path:
                raise SystemExit(
                    f"--backends token '{tok}' has empty name or path"
                )
            if name in seen:
                raise SystemExit(
                    f"--backends: duplicate backend name '{name}'"
                )
            seen.add(name)
            out.append((name, Path(path)))
        return out
    return [("", args.bench)]


_VALID_LATTICES = ("q19", "q27")
_VALID_COLLISIONS = ("bgk", "kbc")


def _resolve_configs(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Normalise --configs / --lattice / --collision into a list of pairs.

    When --configs is supplied, parse each ``LATTICE:COLLISION`` token,
    validate against the same allowed values clipp accepts on the C++ side,
    and reject the only currently-illegal pairing (kbc requires q27 — the
    moment matrix needed by the entropic step is only defined for the full
    27-velocity lattice; the C++ binary would also reject q19+kbc with a
    diagnostic but failing fast in Python avoids a per-run subprocess).
    Empty / malformed tokens raise SystemExit so the user sees the issue
    once instead of N times.
    """
    if args.configs:
        out: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for tok in args.configs:
            if ":" not in tok:
                raise SystemExit(
                    f"--configs token '{tok}' must be LATTICE:COLLISION "
                    f"(e.g. q19:bgk)"
                )
            lattice, _, collision = tok.partition(":")
            lattice = lattice.strip().lower()
            collision = collision.strip().lower()
            if lattice not in _VALID_LATTICES:
                raise SystemExit(
                    f"--configs: unknown lattice '{lattice}' "
                    f"(allowed: {', '.join(_VALID_LATTICES)})"
                )
            if collision not in _VALID_COLLISIONS:
                raise SystemExit(
                    f"--configs: unknown collision '{collision}' "
                    f"(allowed: {', '.join(_VALID_COLLISIONS)})"
                )
            if lattice == "q19" and collision == "kbc":
                raise SystemExit(
                    "--configs: q19+kbc is not supported (KBC entropic "
                    "stabilisation requires the full D3Q27 lattice)."
                )
            pair = (lattice, collision)
            if pair in seen:
                continue  # dedupe duplicates rather than running twice
            seen.add(pair)
            out.append(pair)
        if not out:
            raise SystemExit("--configs supplied but resolved to zero pairs")
        return out
    return [(args.lattice, args.collision)]


def do_sweep(args: argparse.Namespace) -> list[Path]:
    """Run every (backend, config, gpus, size) combination, return paths of written reports."""
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = _resolve_configs(args)
    backends = _resolve_backends(args)
    written: list[Path] = []
    idx = 0
    for backend_name, bench_path in backends:
        for lattice, collision in pairs:
            for overlap in args.overlaps:
                for gpus in args.gpus:
                    for n in args.sizes:
                        spec = RunSpec(
                            n=n,
                            gpus=gpus,
                            lattice=lattice,
                            collision=collision,
                            iters=args.iters,
                            warmup=args.warmup,
                            re=args.re,
                            ulb=args.ulb,
                            label=args.label,
                            backend=backend_name,
                            overlap=overlap,
                        )
                        # Filename includes the backend tag (when present) so
                        # cuda + sycl runs with the same (config, gpus, n)
                        # don't clobber each other in a shared output dir.
                        backend_prefix = (
                            f"{backend_name}-" if backend_name else ""
                        )
                        report_path = (
                            output_dir
                            / f"run-{idx:03d}-{backend_prefix}"
                              f"{spec.lattice}-{spec.collision}-"
                              f"overlap-{spec.overlap}-"
                              f"n{spec.n}-g{spec.gpus}.json"
                        )
                        run_one(bench_path, spec, report_path, dry_run=args.dry_run)
                        if not args.dry_run:
                            written.append(report_path)
                        idx += 1
    print(f"sweep done — wrote {len(written)} report(s) under {output_dir}")
    return written


def load_reports(dirs: list[Path]) -> list[Report]:
    """Recursively load every *.json under ``dirs`` that matches our schema."""
    reports: list[Report] = []
    for d in dirs:
        if not d.exists():
            sys.stderr.write(f"warning: report dir does not exist: {d}\n")
            continue
        for p in sorted(d.rglob("*.json")):
            try:
                with p.open() as f:
                    raw = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                sys.stderr.write(f"warning: skipping {p}: {e}\n")
                continue
            schema = raw.get("schema", "")
            if schema != SCHEMA:
                sys.stderr.write(
                    f"warning: skipping {p}: schema='{schema}' != '{SCHEMA}'\n"
                )
                continue
            cfg = raw.get("config", {})
            sysinfo = raw.get("system", {})
            res = raw.get("results", {})
            gpu_names = [g.get("name", "?") for g in sysinfo.get("gpus", [])]
            # The C++ binary started writing `system.backend` in 2026-05.
            # Older reports lack the field; fall back to "" so plot grouping
            # treats them as a single "untagged" backend (which is what they
            # are for the purposes of legend / colour assignment).
            reports.append(
                Report(
                    n=int(cfg.get("n", 0)),
                    gpus=int(sysinfo.get("gpu_count", 0)),
                    lattice=str(cfg.get("lattice", "?")),
                    collision=str(cfg.get("collision", "?")),
                    label=str(cfg.get("label", "")),
                    backend=str(sysinfo.get("backend", "")),
                    overlap=str(cfg.get("overlap", "none")),
                    mlups=float(res.get("mlups", 0.0)),
                    mlups_per_gpu=float(res.get("mlups_per_gpu", 0.0)),
                    timing_seconds=float(res.get("timing_seconds", 0.0)),
                    timed_iters=int(res.get("timed_iters", 0)),
                    effective_bandwidth_gb_s=float(
                        res.get("effective_bandwidth_gb_s", 0.0)
                    ),
                    gpu_name=gpu_names[0] if gpu_names else "?",
                    raw=raw,
                )
            )
    return reports


def _series_label(r: Report, x_axis: str) -> str:
    """Series key for the legend — everything that varies *between* lines."""
    base = f"{r.lattice} / {r.collision} | overlap={r.overlap}"
    backend_bit = f"{r.backend}, " if r.backend else ""
    if x_axis == "size":
        line = f"{base} | {backend_bit}{r.gpus} GPU{'s' if r.gpus != 1 else ''}"
    elif x_axis == "gpus":
        line = f"{base} | {backend_bit}N={r.n}"
    else:  # x_axis == "config" — bars are already grouped by config; the
        # legend separates the *other* dimensions (backend, N, gpu count,
        # label).
        line = f"{backend_bit}N={r.n} | {r.gpus} GPU{'s' if r.gpus != 1 else ''}"
    if r.label:
        line = f"{line} | {r.label}"
    return line


def _config_key(r: Report) -> str:
    """Stable bar label for ``--plot-x config``: 'q19 / bgk', 'q27 / kbc', …"""
    return f"{r.lattice} / {r.collision}"


def _y_value(r: Report, key: str) -> float:
    return {
        "mlups": r.mlups,
        "mlups_per_gpu": r.mlups_per_gpu,
        "bandwidth": r.effective_bandwidth_gb_s,
    }[key]


def _y_label(key: str) -> str:
    return {
        "mlups": "MLUPS (10\u2076 lattice updates / s)",
        "mlups_per_gpu": "MLUPS / GPU (10\u2076 lattice updates / s)",
        "bandwidth": "Effective HBM bandwidth (GB / s)",
    }[key]


def _plot_lines(args: argparse.Namespace, reports: list[Report]) -> None:
    """``--plot-x size|gpus`` — one curve per series key, MLUPS vs N or GPUs."""
    import matplotlib.pyplot as plt

    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in reports:
        x = float(r.n) if args.plot_x == "size" else float(r.gpus)
        series[_series_label(r, args.plot_x)].append((x, _y_value(r, args.plot_y)))

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, pts) in enumerate(sorted(series.items())):
        pts.sort(key=lambda t: t[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(
            xs, ys,
            marker="o", linewidth=1.6, markersize=5,
            color=color_cycle[i % len(color_cycle)],
            label=name,
        )

    ax.set_xlabel("N (cavity edge length, cells)" if args.plot_x == "size"
                  else "Number of GPUs")
    ax.set_ylabel(_y_label(args.plot_y))
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    if args.plot_x == "gpus":
        ax.set_xticks(sorted({int(x) for pts in series.values() for x, _ in pts}))
    if args.plot_y in ("mlups", "mlups_per_gpu", "bandwidth"):
        ax.set_ylim(bottom=0.0)
    ax.legend(loc="best", fontsize=9, frameon=True)
    gpu_name = reports[0].gpu_name
    title = args.plot_title or (
        f"Carbon LBM lid-driven cavity — {gpu_name}\n"
        f"{'MLUPS vs grid size' if args.plot_x == 'size' else 'Strong scaling'}"
    )
    ax.set_title(title, fontsize=11)
    fig.tight_layout()

    out_path = args.plot_out
    if out_path is None:
        suffix = "vs_size" if args.plot_x == "size" else "vs_gpus"
        out_path = (
            (args.input_dirs[0] if args.input_dirs else args.output_dir)
            / f"mlups_{suffix}.png"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"plot saved to {out_path}")
    if args.show:
        plt.show()
    plt.close(fig)


def _plot_bars(args: argparse.Namespace, reports: list[Report]) -> None:
    """``--plot-x config`` — grouped bar chart, one bar per (lattice, collision).

    Natural pairing with ``--configs`` at fixed (N, gpu_count): three bars
    side by side comparing q19/bgk vs q27/bgk vs q27/kbc, with the y value
    annotated above each bar so the chart is readable on its own.

    When the input mixes multiple (N, gpu_count) per config (e.g. you ran
    ``--sizes 128 256`` alongside ``--configs ...``), bars are grouped by
    config and within each group split by series — same legend key as the
    line plots, just rendered as adjacent bars instead of curve markers.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # configs (x positions) and series (within-group bars).
    configs: list[str] = sorted({_config_key(r) for r in reports})
    series_keys: list[str] = sorted({_series_label(r, "config") for r in reports})
    # values[(config, series)] → list[float]; we average (and surface n) when
    # the user passed duplicate runs of the same cell.
    cell_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in reports:
        cell_values[(_config_key(r), _series_label(r, "config"))].append(
            _y_value(r, args.plot_y)
        )

    n_groups = len(configs)
    n_bars = len(series_keys)
    bar_w = 0.8 / max(n_bars, 1)
    x_centers = np.arange(n_groups, dtype=float)

    fig, ax = plt.subplots(figsize=(max(6.0, 2.0 + 1.6 * n_groups), 4.6))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bottom = 0.0
    for s_idx, s_key in enumerate(series_keys):
        ys = [
            float(np.mean(cell_values[(c, s_key)])) if cell_values[(c, s_key)] else 0.0
            for c in configs
        ]
        offsets = (s_idx - (n_bars - 1) / 2.0) * bar_w
        bars = ax.bar(
            x_centers + offsets, ys, width=bar_w,
            color=color_cycle[s_idx % len(color_cycle)],
            label=s_key if n_bars > 1 else None,
        )
        # annotate each bar with its value (rounded to 1 decimal — MLUPS
        # rarely needs more, bandwidth same).
        for bar, y in zip(bars, ys):
            if y <= 0:
                continue
            ax.annotate(
                f"{y:,.0f}" if y >= 100 else f"{y:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2.0, y),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=9,
            )
        bottom = max(bottom, max(ys, default=0.0))

    ax.set_xticks(x_centers)
    ax.set_xticklabels(configs)
    ax.set_xlabel("Configuration (lattice / collision)")
    ax.set_ylabel(_y_label(args.plot_y))
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    if args.plot_y in ("mlups", "mlups_per_gpu", "bandwidth"):
        # Add ~10% headroom so the value annotations don't clip the top.
        ax.set_ylim(bottom=0.0, top=bottom * 1.12 if bottom > 0 else 1.0)
    if n_bars > 1:
        ax.legend(loc="best", fontsize=9, frameon=True)

    gpu_name = reports[0].gpu_name
    # When every report is at the same (N, gpu_count), surface that in the
    # title — single-knob runs are the common case for this plot mode.
    n_set = sorted({r.n for r in reports})
    g_set = sorted({r.gpus for r in reports})
    fixed_bits = []
    if len(n_set) == 1:
        fixed_bits.append(f"N={n_set[0]}")
    if len(g_set) == 1:
        fixed_bits.append(f"{g_set[0]} GPU{'s' if g_set[0] != 1 else ''}")
    fixed_str = " / ".join(fixed_bits)
    title = args.plot_title or (
        f"Carbon LBM lid-driven cavity — {gpu_name}\n"
        f"MLUPS by configuration"
        + (f" ({fixed_str})" if fixed_str else "")
    )
    ax.set_title(title, fontsize=11)
    fig.tight_layout()

    out_path = args.plot_out
    if out_path is None:
        out_path = (
            (args.input_dirs[0] if args.input_dirs else args.output_dir)
            / "mlups_by_config.png"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"plot saved to {out_path}")
    if args.show:
        plt.show()
    plt.close(fig)


def do_plot(args: argparse.Namespace) -> None:
    dirs = args.input_dirs if args.input_dirs else [args.output_dir]
    reports = load_reports([Path(d) for d in dirs])
    if not reports:
        raise SystemExit("no reports found — nothing to plot")
    if args.plot_x == "config":
        _plot_bars(args, reports)
    else:
        _plot_lines(args, reports)


def main() -> int:
    args = parse_args()
    if args.mode in ("sweep", "sweep+plot"):
        if not args.bench.exists() and not args.dry_run:
            sys.stderr.write(
                f"error: --bench {args.bench} not found. Build the binary first:\n"
                f"  cmake --build build --target carbon_benchmark_lbm_lid_driven_cavity -j\n"
            )
            return 1
        if shutil.which("nvidia-smi") is None:
            sys.stderr.write(
                "warning: nvidia-smi not on PATH; multi-gpu sweeps assume "
                "consecutive device indices via CUDA_VISIBLE_DEVICES.\n"
            )
        do_sweep(args)
    if args.mode in ("plot", "sweep+plot"):
        do_plot(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
