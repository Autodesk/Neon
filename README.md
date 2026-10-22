# carbon

Multi-GPU grid computation library — a clean-room rewrite of
[Neon](https://github.com/Autodesk/Neon).

Same problem domain (dense, multi-GPU, z-partitioned grids; Cartesian stencil compute;
SoA slab layout; host/device views), modernised C++23 surface, two backends (CUDA and SYCL
through AdaptiveCpp), optional MPI+NCCL/RCCL distributed transport, and three on-disk
formats (carbon-native, HDF5, nanoVDB) plus VTK legacy.

## Status

- **46 tests** pass under both backends (40 default + 6 distributed under MPI).
- **4 examples**: `init_and_dense`, plus three canonical unified examples
  (`jacobi_unified`, `cg_poisson_unified`, `salt_pepper_filter`) that each
  cover Dense / DenseDisg / Block × CUDA / SYCL × single-GPU / multi-GPU /
  cross-rank × `OverlapMode::{None,Standard,Extended,TwoWayExtended}`, all
  selected at runtime from one source file.
- **1 benchmark** — `lbm_lid_driven_cavity` (D3Q19/D3Q27 × BGK/KBC) hitting
  ~5130 MLUPS / ~780 GB/s effective HBM on a single A100 80 GB PCIe at N=256.
- **Format-check** wired as a CTest entry (`carbon_format_check`).

## Highlights

- **Singular handles for multi-GPU.** `Buffer<T>`, `Stream`, `Event`, `Field<T, C>` and
  `Halo<T, C>` are presented as one logical object even though each is backed by a
  per-GPU collection. Application code does not branch on GPU count at the call site.
- **`Skeleton` graph-execution layer.** User appends `Kernel`s in topological order;
  `compile()` walks each kernel's `(field_uid, AccessMode, AccessPattern)` tokens, builds
  a DAG, runs a transitive reduction, performs greedy multi-stream scheduling, and
  optionally rewrites the graph for halo+stencil overlap (`OverlapMode::None /
  Standard / Extended / TwoWayExtended`). The compile pass also auto-inserts halos
  before any stencil read of a dirty field — multi-GPU correctness with no manual halo
  plumbing.
- **Universal kernels.** A single `CARBON_HOST_DEVICE` lambda compiles to both the GPU
  launch path and an OpenMP host loop; the polymorphic `Kernel::launch_on_device` /
  `launch_on_host` selectors give in-process device/host parity testing on the same
  body.
- **Two backends, one source.** `CARBON_BACKEND={Cuda, Sycl}` is fixed at carbon-build
  time. Every `.cu` test compiles under both nvcc (CUDA TU) and AdaptiveCpp acpp
  (SYCL TU); the test list is single-source.
- **Three transports, one API.** Single-process, single-process multi-GPU, and
  one-rank-per-GPU under MPI+NCCL (or +RCCL for AMD HIP targets). MPI/NCCL/RCCL stay
  PRIVATE-linked into `libcarbon.so`; consumer TUs never include `<mpi.h>` or
  `<nccl.h>`.

## Layout

```
include/carbon/              Public API (umbrella: <carbon/carbon.h>)
  detail/                    Internal helpers (kept off the user surface)
    backend/{cuda,sycl}/     Per-backend stream / event / GPU / kernel-launch glue
    dense/                   Dense-grid primitives: idx, span, partition, field
    skeleton/                Skeleton implementation .inl files
src/                         Library sources + carbon::io implementations
tests/                       CTest unit tests (host .cpp + device .cu)
examples/                    Didactic single-concept programs
benchmarks/                  Headline performance workloads
docs/                        Hand-curated docs (Skeleton DOT references, design notes)
cmake/                       FetchContent + Find* helpers
scripts/                     Maintenance scripts (clean-builds, docker-build-all)
docker/                      Per-config Dockerfiles (cuda / sycl / sycl-rccl)
third_party/                 Vendored single-header libs (clipp)
AGENTS.md                    Operational state (current architecture + build matrix)
docs/CHANGELOG.md            Append-only history of design decisions
```

## Build

Requirements:

- CMake ≥ 3.20
- A C++23 host compiler
- One compute backend toolchain:
  - **CUDA** — CUDA Toolkit (tested with recent nvcc; `.cu` TUs use C++20).
  - **SYCL** — [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) ≥ 24.x with
    a CUDA or HIP target installed.
- OpenMP

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Backend selection

```bash
# CUDA (default)
cmake -S . -B build -DCARBON_BACKEND=Cuda

# SYCL via AdaptiveCpp; cuda:sm_80 target + OMP host fallback
cmake -S . -B build-sycl \
  -DCARBON_BACKEND=Sycl \
  -DCARBON_SYCL_TARGETS="cuda:sm_80;omp" \
  -DCMAKE_CXX_COMPILER=/path/to/clang++-19
```

`CARBON_SYCL_TARGETS` follows AdaptiveCpp's `ACPP_TARGETS` syntax (e.g. `cuda:sm_90`,
`cuda:sm_80;omp`, `omp`, `hip:gfx90a`). `CARBON_SYCL_CUDA_PATH` /
`CARBON_SYCL_ROCM_PATH` forward toolkit paths to acpp when the auto-detected default
isn't right.

### Optional transports

| Flag                            | Adds                                                           |
| ------------------------------- | -------------------------------------------------------------- |
| `-DCARBON_ENABLE_MPI=ON`        | `Distributed` + `DistTransfer<T>`; NCCL on CUDA targets, RCCL on HIP. Adds 6 distributed CTest entries. |
| `-DCARBON_ENABLE_HDF5=ON`       | `carbon::io::write_hdf5` / `read_hdf5`. Parallel HDF5 (collective MPI-IO) auto-detected when paired with `CARBON_ENABLE_MPI`. |
| `-DCARBON_ENABLE_NANOVDB=ON`    | `carbon::io::write_nanovdb` / `read_nanovdb`. Discovery: explicit dir → installed `nanovdb` config → FetchContent of OpenVDB pinned to `CARBON_OPENVDB_GIT_TAG`. |

VTK legacy I/O (`carbon::io::write_vtk` / `read_vtk`) is unconditionally available —
pure stdlib, no external dependency.

### Build options

| Option                          | Default | Effect                                                       |
| ------------------------------- | :-----: | ------------------------------------------------------------ |
| `CARBON_BUILD_TESTS`            |   ON    | Build the unit test tree.                                    |
| `CARBON_BUILD_EXAMPLES`         |   ON    | Build the example programs.                                  |
| `CARBON_BUILD_BENCHMARKS`       |   ON    | Build the benchmark programs.                                |
| `CARBON_ENABLE_MPI`             |   OFF   | Distributed transport (MPI + NCCL/RCCL).                     |
| `CARBON_ENABLE_HDF5`            |   OFF   | HDF5 read/write of `Dense` fields.                           |
| `CARBON_ENABLE_NANOVDB`         |   OFF   | nanoVDB read/write of `Dense` fields.                        |
| `CARBON_BACKEND`                |  Cuda   | Compute backend: `Cuda` or `Sycl`.                           |

## Tests

```bash
ctest --test-dir build --output-on-failure
```

Default registers 40 tests + the `carbon_format_check` style guard (when `clang-format`
is on PATH). With `CARBON_ENABLE_MPI=ON`, six `mpiexec`-driven distributed tests join
under the `distributed` CTest label. See [`tests/README.md`](tests/README.md) for the
per-test scope (CPU-only / Multi-GPU / Distributed multi-GPU).

The Docker matrix script `scripts/docker-build-all.sh` builds and validates every
backend × transport configuration in a dedicated container; ideal for
pre-push hygiene on a host that lacks (e.g.) ROCm or AdaptiveCpp.

```bash
scripts/docker-build-all.sh                 # all five configs
scripts/docker-build-all.sh --only sycl-rccl # AMD-only path
```

## Examples & benchmarks

```bash
# Same per-cell body, every grid family, every transport, every overlap mode
./build/examples/carbon_example_jacobi_unified --grid dense --iters 200
./build/examples/carbon_example_jacobi_unified --grid disg  --iters 200 --overlap standard
./build/examples/carbon_example_jacobi_unified --grid block --iters 200 --overlap two-way-extended
mpiexec -n 2 ./build/examples/carbon_example_jacobi_unified --grid dense --overlap standard

# Lid-driven cavity LBM (single GPU, scrape MLUPS=... line on stdout)
./build/benchmarks/lbm_lid_driven_cavity/carbon_benchmark_lbm_lid_driven_cavity \
    --n 256 --iters 1000 --warmup 200

# LBM under MPI (one rank per GPU)
mpiexec -n 4 ./build/benchmarks/.../carbon_benchmark_lbm_lid_driven_cavity \
    --n 512 --iters 2000 --warmup 200
```

The LBM benchmark also emits a structured JSON report (`--report PATH`) consumable by
`benchmarks/lbm_lid_driven_cavity/scripts/sweep_and_plot.py`, which renders MLUPS-vs-N
or strong-scaling figures.

## Use from another project

Carbon installs a `find_package(carbon)`-friendly package config:

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/path/to/prefix
cmake --build build -j
cmake --install build
```

In the consumer's `CMakeLists.txt`:

```cmake
find_package(carbon REQUIRED)
target_link_libraries(my_app PRIVATE carbon::carbon)
```

`carbonConfig.cmake` calls `find_dependency` for OpenMP / CUDAToolkit / spdlog;
MPI / NCCL / RCCL stay PRIVATE-linked, so an `CARBON_ENABLE_MPI=ON` install does not
require the consumer to discover them.

## Code style

Formatted with `clang-format` 20+. The config encodes the "divided" signature style
(return type / qualifiers on one line; function name on the next; one parameter per
line; member-initializer lists always stacked). `Neon/` is excluded via
`.clang-format-ignore`.

```bash
# Rewrite in-place
cmake --build build --target format

# Non-zero on drift (CI-friendly)
cmake --build build --target format-check
```

A pre-commit config (`.pre-commit-config.yaml`) is provided; `pip install --user
pre-commit && pre-commit install` wires the same hook into `git commit`.

## API documentation

Doxygen-rendered HTML site (when `doxygen` is on PATH):

```bash
cmake --build build --target docs
xdg-open build/docs/html/index.html
```

The Doxyfile lives at [`cmake/Doxyfile.in`](cmake/Doxyfile.in); `README.md` is the
main page, [`AGENTS.md`](AGENTS.md) is reachable from the sidebar, and the `docs/`
folder is included verbatim.

## Contributing / agents

[`AGENTS.md`](AGENTS.md) is the operational source of truth — current architecture,
build matrix, conventions, status snapshot, and the "next big features" list.
Long-form design rationale lives under [`docs/CHANGELOG.md`](docs/CHANGELOG.md) (the
project memory; append-only).

## License

Apache License 2.0 — same as the upstream Neon project. See
[`LICENSE`](LICENSE) for the full text and the third-party attributions
(spdlog, RapidJSON).

Copyright 2026 Autodesk, Inc.
