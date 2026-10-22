# AGENTS.md

Operational context for any agent (human or AI) working in this repository.

## Mission

`carbon/` is a step-by-step, clean-architecture rewrite of the
[Neon](https://github.com/Autodesk/Neon) multi-GPU grid library. The upstream sources are vendored
read-only under `Neon/` as the reference implementation; do not modify them. All new code lives in
the `carbon::` namespace and under `include/carbon/`, `src/`, `tests/`.

The current focus is a port of Neon's `dGrid` (dense, multi-GPU, Z-partitioned grid) data plane.

## Repository layout

```
include/carbon/              Public API (umbrella: carbon/carbon.h)
  detail/                    Internal: GPU/buffer/stream/event wrappers
  detail/dense/              Dense-grid primitives: idx, span, partition, field, kernel, launch_shape
  boundary_mode.h            BoundaryMode enum (Open / PeriodicZ), both honoured end-to-end
  data_view.h                DataView enum (Standard / Internal / Boundary)
  devices.h                  carbon::get_all_devices() free function
  distributed.h              MPI handle (gated by CARBON_HAVE_MPI_NCCL)
  dist_transfer.h            DistTransfer<T> NCCL Send/Recv (gated)
  halo.h                     Halo<T, Components> intra-process Transfer + cross-rank DistTransfer (Open + PeriodicZ)
  io.h                       carbon::io::{write,read}_native + write/read_vtk (legacy) + (gated) write/read_hdf5 + (gated) write/read_nanovdb
  integer3d.h, floating3d.h  3D tuples + aliases (int8_3d, int32_3d, int64_3d, float32_3d, float64_3d)
src/                         Library implementation + CUDA stub
tests/                       CTest unit tests (host .cpp and device .cu); see tests/README.md
examples/                    Opt-in example programs (CARBON_BUILD_EXAMPLES, ON by default)
benchmarks/                  Opt-in performance workloads (CARBON_BUILD_BENCHMARKS, ON by default); see benchmarks/README.md
third_party/                 Vendored single-header libs used only by examples (clipp; matches Neon's bundled CLI parser)
cmake/                       FetchContent glue (spdlog, rapidjson) + FindNCCL.cmake
Neon/                        Read-only upstream reference
.clang-format(-ignore)       Style enforcement (Neon/** excluded)
```

Optional MPI build adds `-DCARBON_ENABLE_MPI=ON`; MPI and NCCL are PRIVATE-linked into
`libcarbon.so` so consumer TUs never see `<mpi.h>` or `<nccl.h>` (umbrella branches on
`CARBON_HAVE_MPI_NCCL` to expose `<carbon/distributed.h>` / `<carbon/dist_transfer.h>`).

Optional HDF5 build adds `-DCARBON_ENABLE_HDF5=ON`; HDF5 is PRIVATE-linked and the
`<carbon/io.h>` HDF5 templates are guarded by `CARBON_HAVE_HDF5`. When `CARBON_ENABLE_MPI`
is also set and the discovered HDF5 was compiled with MPI support, `CARBON_HAVE_PHDF5` is
defined and `write_hdf5` / `read_hdf5` use collective MPI-IO writes against a single file.

VTK legacy I/O is always available — no external dependency, pure stdlib.
`carbon::io::write_vtk` / `read_vtk` write `DATASET STRUCTURED_POINTS` BINARY files
with one `SCALARS name dtype N` block per field under a shared `POINT_DATA`
section, big-endian payload (legacy spec mandates network byte order), component-
fastest interleaving. Multi-rank uses one file per rank with `.rank<N>` suffix —
each shard is a self-contained VTK that ParaView/VTK can open independently.
Reader is strict (rejects XML VTK / non-BINARY / non-STRUCTURED_POINTS / non-unit
spacing / mismatched DIMENSIONS) so a wrong file fails fast rather than scrambling
field data.

Optional nanoVDB build adds `-DCARBON_ENABLE_NANOVDB=ON`. Discovery order: first
`-DCARBON_NANOVDB_INCLUDE_DIR=/path/to/nanovdb/headers` if the user already has
nanoVDB on disk, then `find_package(nanovdb CONFIG)` for system-installed packages,
falling back to a `FetchContent` of OpenVDB pinned to `v11.0.0` (override with
`-DCARBON_OPENVDB_GIT_TAG=...`) — the populate-only path skips OpenVDB's top-level
CMake (no TBB/Boost/blosc dependency) and just exposes the `nanovdb/` include
directory. nanoVDB is PRIVATE-linked and the `<carbon/io.h>` nanoVDB templates are
guarded by `CARBON_HAVE_NANOVDB`. `write_nanovdb` packs each `(field, component)` as
one `nanovdb::build::Grid<T>` and flushes them as a multi-grid `.nvdb` via
`nanovdb::io::writeGrids`; `read_nanovdb` mirrors via `readGrids` + per-grid
`getAccessor()`. Multi-rank writes use one file per rank (`.rank<N>` suffix) since
nanoVDB has no parallel writer. Supported scalar types in this pass: `float`,
`double`, `int32_t`. Single-component fields get one grid named exactly `"field_<i>"`;
multi-component fields fan out into `"field_<i>_c<c>"` grids.

## Compute backend (CUDA or SYCL)

Carbon supports two compute backends, selected at carbon-build time via
`-DCARBON_BACKEND=Cuda` (default) or `-DCARBON_BACKEND=Sycl`. Both backends share the
same public API; what differs is which `detail/backend/<name>/` headers are pulled in
and which toolchain compiles the per-cell device code.

- **CUDA** is the historical and feature-complete backend. `nvcc` compiles `.cu` TUs;
  per-cell launches go through `cudaLaunchKernel`. 90/90 default tests pass and all
  example programs build and run.
- **SYCL** is supported through [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp)
  (a.k.a. `acpp`, formerly hipSYCL). Per-cell launches go through
  `sycl::queue::parallel_for`; reductions through `sycl::reduction` over
  `q.parallel_for` (single-stage, no per-block partials); cross-device peer copies via
  a process-wide shared `sycl::context` (so `q.memcpy(dst_gpu_ptr, src_gpu_ptr, n)` does
  the right thing whether `src`/`dst` sit on the same or different devices). 89/89
  default SYCL tests pass with cross-backend numerical parity. The 4-test delta vs
  CUDA is exactly the `carbon_add_cuda_only_unit_test` gate — the explicit opt-out
  for tests that need raw CUDA-runtime APIs in their *body*: `block_cuda` (a smoke
  test pinning the `cudaMalloc` / `cudaMemset` / `cudaDeviceSynchronize` /
  `cudaGetErrorString` boundary itself) plus `mgrid_cross_level`, `mgrid_multi_gpu`
  and `mgrid_weighted_cuts` (hand-rolled `<<<>>>` kernels — mGrid's `KernelLoader`
  integration landed after those tests). Everything else, including every
  distributed test, is registered on both backends.

  SYCL build (needs AdaptiveCpp + clang; adjust the compiler paths below to wherever your
 install lives — the `/home/max/.pixi/envs/adaptivecpp` prefix these commands used to name
 no longer exists on the dev host, so treat it as an example rather than a working path):
  ```bash
  cmake -S . -B build-sycl \
      -DCARBON_BACKEND=Sycl \
      -DCARBON_SYCL_TARGETS="cuda:sm_80;omp" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=/home/max/.pixi/envs/adaptivecpp/bin/clang++-19
  cmake --build build-sycl -j
  ctest --test-dir build-sycl --output-on-failure
  ```
  `CARBON_SYCL_TARGETS` follows AdaptiveCpp's `ACPP_TARGETS` syntax (e.g. `cuda:sm_90`,
  `cuda:sm_80;omp`, `omp` only). `CARBON_SYCL_CUDA_PATH` is forwarded to `acpp` as
  `--cuda-path` and to the linker as `-L$path/lib64`. Note that adding
  `-DCARBON_ENABLE_MPI=ON` on top of this does not configure on the dev box: the local
  MPI wrappers are built for g++ and `find_package(MPI)` fails its `MPI_CXX_WORKS`
  compile check under the pixi clang. SYCL+MPI is validated in the container matrix
  (`scripts/docker-build-all.sh --only sycl-mpi`), not natively here.

Adding `-DCARBON_ENABLE_MPI=ON` registers **26** further distributed tests (ctest
label `distributed`, one file per grid family × transport concern) on either
backend, for **116** total on CUDA. They cover intra-process multi-GPU + cross-rank
for every grid family, the `OverlapMode` matrix on every grid family
(`skeleton_dist_overlap{,_disg,_block}`), multi-format I/O round-trip (`io_dist`),
non-`Sum` reduction shapes on every grid family (`*_dist_reduce_ops`), and the
Dense-checkpoint atomic-commit / manifest read path (`checkpoint_dense_dist`).
`ctest --test-dir <build> -N -L distributed` prints the current list.

The backend dispatcher lives at `include/carbon/detail/backend/select.h` and exposes
`CARBON_BACKEND_IS_CUDA` / `CARBON_BACKEND_IS_SYCL` / `CARBON_BACKEND_IS_HIP` plus the
folded `CARBON_BACKEND_IS_CUDALIKE` (CUDA *or* HIP) and `CARBON_CUDALIKE_DEVICE_TU`
(`__CUDACC__` *or* `__HIPCC__`) macros for the backend forks (stream sync in
`detail/stream_group.h::wait()`, `System::async_copy`, the launch bodies in
`dense_kernel_impl.h` / `reduce_kernel_impl.h`). Backend-specific impls live under just
**two** trees: `detail/backend/cudalike/` (shared by the CUDA and HIP backends — it
talks to the runtime only through the neutral `gpu_*` aliases + `gpucub` namespace in
`detail/backend/cuda_like.h`, so one source tree serves `cudaXxx`+`cub` and `hipXxx`+
`hipcub`, including HIP-on-NVIDIA) and `detail/backend/sycl/`. The legacy paths
(`detail/stream.h`, `detail/event.h`, `detail/gpu.h`, `detail/dense/launch.h`, etc.)
are dispatch shims that resolve to the matching backend file.

## Build & verify (always run all three after edits)

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --build build --target format-check
```

> **Pass `-DCMAKE_CUDA_ARCHITECTURES` explicitly; the CMake default can break every launch.**
> The repo deliberately does not pin an arch, so an unadorned `cmake -S . -B build` takes
> CMake's default — `75` on this host. Nothing warns you, the build is clean, and then *every*
> kernel launch fails with `the provided PTX was compiled with an unsupported toolchain`
> (`gpu error 222`), because no `sm_89` cubin exists and the driver must JIT the embedded PTX.
> That JIT is what fails: `/usr/local/cuda` is **13.1** while driver 580.178.04 is a **CUDA
> 13.0** driver, and a 13.0 driver cannot consume 13.1 PTX. Verified 2026-08 by a 2x2 probe —
> 13.1/`sm_75` fails, and 13.1/`sm_89`, 13.0/`sm_75`, 13.0/`sm_89` all pass — so it takes the
> *combination* of the newer toolkit and a non-native arch. It cost 53 of 128 tests on a tree
> that had been in that state since 2026-08-19. Configure with
> `-DCMAKE_CUDA_ARCHITECTURES=89` on this box (native, no JIT, and faster); building against
> `/usr/local/cuda-13.0` also works. If you see mass `gpu error 222`, check the arch before
> reading any further.
>
> **Run `ctest` on >= 2 GPUs before believing a green result.** The two-GPU path used to fail
> 15 tests that the single-GPU suite passed. Root-caused 2026-07 and **fixed**: on a platform
> whose driver reports P2P capability that does not work (Ada over PCIe here), a cross-device
> `cudaMemcpyAsync` returns `cudaSuccess` and transfers **nothing** once peer access is
> enabled, so halos delivered stale cells and reductions dropped a GPU's partials with no
> error anywhere. `GpuGroup` now gates peer access on an NVLink-class link check plus a
> round-trip delivery probe. See `docs/CHANGELOG.md` "2026-07 — Pre-feature audit".
>
> **Peer access policy — `CARBON_P2P`.** `auto` (default) enables a direction only if the link
> reports NVLink-class capability *and* a delivery probe passes; `off` never enables it (all
> cross-device copies host-staged, always correct); `on` enables everything the driver claims,
> subject only to the probe. On a PCIe-P2P platform that is genuinely healthy, `auto` costs
> bandwidth — use `CARBON_P2P=on` there. Correctness never depends on P2P: it is a
> performance path, and the host-staged fallback is the correct baseline.
> The policy lives in `include/carbon/detail/backend/cudalike/peer_access.h` and is shared by
> the **CUDA and HIP** backends (HIP previously enabled no peer access at all). Its
> NVLink-class check is an NVIDIA-derived heuristic and is **unverified on ROCm** — if AMD
> reports no native-atomic support for an xGMI pair, HIP keeps the host-staged path it already
> used, so `auto` cannot regress it, only fail to speed it up.
>
> **SYCL is measurably not exposed**, so it has no policy of its own: built and run on the
> affected two-GPU host it passes 89/89 including all 27 `needs_multi_device` tests, and the
> CUDA defect reproduces at idle, so AdaptiveCpp is evidently not taking the peer path.
>
> **`CARBON_P2P=on` is genuinely unsafe on an affected platform, and the probe alone will not
> save you.** Repeated forced-on runs there are non-deterministic: one revoked a single
> direction while the other passed the probe and then corrupted the next test, and three runs
> afterwards passed both directions cleanly. The conservative default is the thing that keeps
> results correct.
>
> **Known open:** `dense_disg_overlapped[Skeleton]` still fails ~10/15 under *artificial*
> concurrent GPU load (clean idle, clean in the normal suite) with a small drift at the
> boundary plane, while `[OverlappedStencil]` passes on the same data. The changelog records
> which fan-in hypotheses were tried and ruled out — read it before re-deriving them.
>
> **Test labels.** `ctest -L cpu` selects the tests that do real work with zero visible
> devices (the CPU CI lane); `-L gpu` needs at least one device; `-L needs_multi_device`
> lists the ~28 tests that self-skip below two GPUs; `-L warp` lists the 31 Python/Warp
> tests (`-DCARBON_ENABLE_PYTHON=ON`). Set `CARBON_TEST_STRICT=1` to make those self-skips
> hard failures (`carbon_multi_gpu_gate`, `carbon_warp_gate`) instead of passes.
>
> **The Python/Warp layer needs no manual setup to run under `ctest`, and did not always
> report honestly.** Each `test_warp_*.py` self-skips when the vendored fork is not
> importable, and a unittest skip is a ctest pass, so until 2026-08 `ctest -R warp` reported
> 20/20 green in 0.65s having compiled nothing — ctest's interpreter is the project venv,
> which has no `warp` installed. Registration now injects
> `PYTHONPATH=<root>/warp:<root>/py` per test, so plain `ctest` exercises the layer for real
> (~50s, 31 tests). If you see the whole warp label finish in under a second, it skipped:
> run `ctest -L warp` with `CARBON_TEST_STRICT=1` and `carbon_warp_gate` will say what is
> missing. Nine of those files were also never registered at all before 2026-08, so the
> pytest path (`scripts/py-test-cuda12.sh`, which globs the directory) and the ctest path
> covered different sets — keep new `test_warp_*.py` files added to `tests/CMakeLists.txt`
> via `carbon_add_warp_test()`.
>
> **Assertions.** Test TUs compile with `-UNDEBUG` so `<cassert>` is live even in a `Release`
> build. Before 2026-07 they did not, which made 305 assertion sites inert — 7 test files had
> no other failure mechanism at all.
>
> **Sanitizers.** `-DCARBON_SANITIZE=address+undefined` (also `address`, `undefined`,
> `thread`) instruments carbon and everything linking it. Host CXX only; use
> `compute-sanitizer` for device-side checks.

`clang-format` 20+ must be on PATH (the `.clang-format` `BinPackParameters:
AlwaysOnePerLine` option needs clang-format >= 20). `pip install --user "clang-format>=20"`
works; then
`export PATH="$HOME/.local/bin:$PATH"`. Use `cmake --build build --target format` to rewrite in
place.

### Docker matrix (all eight configurations from one host)

`scripts/docker-build-all.sh` builds and validates every backend/transport
configuration in a dedicated container so a host that lacks (e.g.) ROCm or
AdaptiveCpp can still verify the SYCL+RCCL and native-HIP build paths. The five
Dockerfiles under `docker/` cover:

- `docker/Dockerfile.cuda` → `carbon-cuda:local` (CUDA + NCCL + OpenMPI on the
  `nvidia/cuda:12.6.3-devel-ubuntu22.04` base) — services the **cuda** and
  **cuda-mpi** configs.
- `docker/Dockerfile.sycl` → `carbon-sycl:local` (same CUDA base + AdaptiveCpp
  25.02 from conda-forge with clang-19/lld-19/llvm-tools-19) — services the
  **sycl** and **sycl-mpi** configs (NVIDIA target via `cuda:sm_XX`).
- `docker/Dockerfile.sycl-rccl` → `carbon-sycl-rccl:local`
  (`rocm/dev-ubuntu-22.04:6.4.4-complete` + AdaptiveCpp from conda) —
  services the **sycl-rccl** config (AMD target via `hip:gfx<arch>` with
  `-DCARBON_SYCL_ROCM_PATH=/opt/rocm`).
- `docker/Dockerfile.hip` → `carbon-hip:local`
  (`rocm/dev-ubuntu-22.04:6.4.4-complete`, no AdaptiveCpp/conda layer) —
  services the **hip** and **hip-mpi** configs of the native HIP backend
  (`-DCARBON_BACKEND=Hip`, AMD target via `-DCARBON_HIP_ARCH=gfx<arch>`,
  RCCL transport on **hip-mpi**). The AMD analogue of the cuda image: the
  `.cu` device TUs compile through CMake's HIP language (amdclang++ from
  `/opt/rocm/bin`), host `.cpp` stays on g++.
- `docker/Dockerfile.hip-nvidia` → `carbon-hip-nvidia:local`
  (`nvidia/cuda:12.6.3-devel-ubuntu22.04` + HIP headers/cmake from the ROCm
  apt repo) — services the **hip-nvidia** config: the native HIP backend with
  `-DCMAKE_HIP_PLATFORM=nvidia`, where the HIP language is compiled by **nvcc**
  and `<hip/hip_runtime.h>` forwards every `hipXxx` to the matching `cudaXxx`
  (hipcub → cub). This is how the HIP source tree gets **runtime** coverage on
  an NVIDIA card — `hip-nvidia` runs the full ctest suite when an NVIDIA GPU is
  present (89/89 on 2x RTX 6000 Ada, `sm_89`; previously 67/67 on an A100,
  `sm_80`). MPI is off for this config (the
  distributed collective auto-selects RCCL for the HIP backend, wrong on
  NVIDIA; HIP-on-NVIDIA + NCCL is a tracked follow-up).

The orchestrator detects local GPUs by probing `nvidia-smi -L` (NVIDIA) and
`/dev/kfd` + `/dev/dri` (AMD). For each config it always runs the cmake +
build + (cuda / hip configs only) `format-check` cycle, and additionally runs
`ctest --output-on-failure` if the matching GPU is reachable from inside the
container (NVIDIA: `--gpus all` via the nvidia-container-toolkit; AMD:
`--device=/dev/kfd --device=/dev/dri --group-add video`). Logs land in
`docker-logs/run-<config>.log` and the build trees are mirrored under
`docker-logs/<config>/build/` for post-mortem inspection. Common knobs:
`--only hip`, `--only sycl-rccl`, `--sm-arch sm_89`, `--gfx-arch gfx90a`,
`--no-cache`, `--skip-build-images`.

### CI workflows

Two GitHub Actions workflows live under `.github/workflows/`:

- **`ci.yml` — CPU-only fast lane.** Runs on every push / PR on hosted
 Ubuntu runners. Three jobs (`format-check`, `cuda-cpu-only`,
 `sycl-cpu-only`) exercise the full configure / build / link path for
 the CUDA and SYCL backends. `cuda-cpu-only` runs `ctest -L cpu`; the
 SYCL job runs the full test set, because with an `omp` target the
 OpenMP host is itself an AdaptiveCpp device. No GPU touched. This is
 the lane every contributor's PR rides through, and its value is the
 configure/build/link gate plus `-DCARBON_WERROR=ON` (the libcarbon
 host compile is warning-clean on every PR) more than test breadth:
 only 9 host tests do real work without a device. It previously
 selected by name exclusion, which had gone stale to the point that
 26 device-requiring tests were in scope and the lane could not have
 been passing — see `docs/CHANGELOG.md` 2026-07.

- **`gpu-ci.yml` — GPU validation lane (P1.γ).** Runs on a **self-hosted**
  runner labelled `[self-hosted, linux, gpu, cuda]` and is triggered by
  (a) the `gpu-ci` label on a PR or (b) `workflow_dispatch` (manual
  re-run from the Actions tab) or (c) pushes to `main` / `master`. The
  workflow shells out to `scripts/docker-build-all.sh --only cuda` and
  `scripts/docker-build-all.sh --only cuda-mpi` so the runner host only
  needs `docker` + `nvidia-container-toolkit` + an NVIDIA GPU — no
  native clang / CUDA / MPI / NCCL / HDF5 install required. Build /
  test logs land as workflow artifacts (`gpu-ci-cuda-logs`,
  `gpu-ci-cuda-mpi-logs`) for 14 days. The matrix covers the two audit
  classes the CPU lane cannot exercise: `cuda` runs the full default
  test suite + `format-check`; `cuda-mpi` adds every
  `distributed`-labelled test (`ctest -N -L distributed` for the current
  list). The workflow
 passes `--werror` (gates the libcarbon host compile on the container
 g++) and its preflight prints the visible GPU count. On a runner with
 `>= 2` GPUs it exports `CARBON_TEST_STRICT=1` (forwarded into the
 container by `scripts/docker-build-all.sh`), so `carbon_multi_gpu_gate`
 fails if the devices turn out not to be visible to the test process —
 a hidden card, MIG partitioning, or a stale `CUDA_VISIBLE_DEVICES`
 no longer downgrades ~28 `needs_multi_device` tests to silent
 no-op passes. Below 2 GPUs it emits a `::warning::` instead, since
 that configuration legitimately covers only the build plus the
 single-GPU suite.

#### Self-hosted GPU runner setup

To turn `gpu-ci.yml` on for your fork:

1. Provision a Linux host with an NVIDIA GPU (Ampere or newer
   recommended for `sm_80` default), Docker, and the
   `nvidia-container-toolkit` configured for `--gpus all`.
2. Register a GitHub Actions self-hosted runner on the repo
   (Settings → Actions → Runners → New self-hosted runner). Apply
   the labels `self-hosted`, `linux`, `gpu`, `cuda` exactly as
   listed; the workflow's `runs-on` matches all four.
3. (Optional) Set the `GPU_CI_SM_ARCH` repo / org Actions variable
   to the runner's compute capability (`sm_86` for Ampere consumer,
   `sm_89` for Ada, `sm_90` for Hopper). Defaults to `sm_80`.
4. (Optional) Add a `gpu-ci` label to your repo. Applying it to a
   PR auto-triggers the GPU lane on that PR; removing it stops
   subsequent re-runs from queuing GPU time.

A fork that has not registered a matching runner sees `gpu-ci.yml`
jobs queue with no available runner — the rest of CI proceeds; the
GPU lane just stays pending. That is the intentional no-op state:
GPU validation is **opt-in** per fork, not a blocker for forks
that don't run GPUs.

## Status snapshot

_Compact feature inventory. **The long-form rationale for each line lives in
[`docs/CHANGELOG.md`](docs/CHANGELOG.md)** — when you need to know exactly
how a feature behaves at the edges, grep the changelog for the section name
(every bullet here corresponds to one or more dated changelog entries)._

### Core plumbing

- **Host bootstrap** — `carbon::init / device_count / info`, spdlog +
  rapidjson, `carbon::get_all_devices()` for the canonical
  "every-visible-GPU" handle.
- **Multi-GPU primitives** — `detail::Gpu`, `GpuGroup`, `BufferGroup`,
  `EventGroup`, `StreamGroup` (per-GPU + N×N fan-in `wait_for`),
  `System` (shared-ptr-backed multi-GPU state, optional `Distributed` +
  `Communicator`, central `async_copy` / `async_memset` byte primitives,
  and `default_stream()` — one lazily-created shared main stream per
  `System`, the anchor every `Skeleton` schedules on by default).
- **Geometry types** — `Stencil` (3D offset descriptor, `int8_3d` storage,
  cached Chebyshev radius; ctor takes `int32_3d` for ergonomic literals),
  `Integer3d<T>` / `Floating3d<T>` templates with `int8/32/64_3d` and
  `float32/64_3d` aliases.

### Grid families (port targets from Neon's `dGrid` / `bGrid`)

- **`Dense<T, C>` — Cartesian z-slab grid.** `Span` carries the haloed
  buffer extent + pitch + halo thickness; `Idx` is purely local
  `(x, y, z)`; `Partition` is a non-owning device-visible view over the
  field storage. `Field` holds Device / Host / HostDevice buffers per
  slab. `BoundaryMode { Open, PeriodicZ }` honoured end-to-end.
  Distributed-aware: when `system.distributed() != nullptr` the global z
  extent is partitioned across `world_size`; `global_slab_count() ==
  world_size`, single-process collapses to `gpu_count()`.
- **`DenseDisg<T, C>` — disaggregated dense (lattice-Boltzmann shape).**
  Same Cartesian domain as `Dense` but components are stored as separate
  z-extruded regions (boundary / owned / ghost) so a lattice-Boltzmann
  stream step touches only the populations it needs. Shares the
  `if constexpr (DisgPolicy)` arm of the dense kernel impls; carries its
  own `Span` / `Idx` / `Partition` with the same duck-typed API.
- **`Block<SBlock>` — block-sparse (port of Neon's `bGrid`).** Bounded
  Cartesian domain chunked into `SBlock::MemX × MemY × MemZ` bricks
  (default `StaticBlock<8, 8, 8>`). Retained bricks carry a packed
  `BitMask` of active voxels; cross-brick reads go through a 27-way
  connectivity table on `Span`. Per-GPU partition splits the brick
  z-axis; multi-GPU + MPI reserve ghost-brick slots at every slab's
  lower/upper edge. `Block::Field::make_halo_kernel()` builds a real
  `HaloBlock<T, C, SBlock>`-backed `Kernel`; the back-pointer set by
  `Block::newField` makes this work transparently for user code.
- **`DataView { Standard, Internal, Boundary }`** — every grid splits its
  iteration domain into the inner band (no halo dependency) and the
  one-halo-thick boundary band. `Internal + Boundary == Standard`
  cell-by-cell, locked by `test_data_view_split` / `test_block_data_view`.
  `Dense` rejects `local_z < 2 * halo_z` so the bands cannot collide.

### Transfer + halo

- **`Transfer<T>` / `DistTransfer<T>`** — host↔device, peer-async via
  `System::async_copy` (`cudaMemcpyPeerAsync` on different GPUs); the
  distributed variant uses grouped NCCL `Send` / `Recv` under
  `CARBON_HAVE_MPI_NCCL`.
- **`Halo<T, C>` (Dense), `HaloDisg<T, C>` (DenseDisg),
  `HaloBlock<T, C, SBlock>` (Block).** Each builds a single intra-process
  `Transfer<T>` over the field's per-GPU allocations plus a cross-rank
  `DistTransfer<T>` companion. All three handle `BoundaryMode::PeriodicZ`
  on Dense / DenseDisg; Block has no global `BoundaryMode` flag (boundary
  behaviour is encoded as which bricks exist). `launch_on_host` is the
  host MPI fallback follow-up.

### Kernel layer

- **`Kernel`** — type-erased polymorphic handle. Six `KernelKind`s:
  `Compute`, `Reduction`, `Halo`, `ComputeHost`, `ReductionHost`,
  `Universal{,Reduction}`. Three factory classes:
  - **Device-only** (`Dense::make_kernel` / `Block::make_kernel` / ...).
  - **Host-only** (`Dense::make_host_kernel` — 3-deep `omp parallel for
    collapse(3)` over `iter_extent(view)`).
  - **Universal** (`Dense::make_universal_kernel`) — both paths from one
    loading lambda; default `launch` forwards to device.
  A heterogeneous `std::vector<Kernel>` iterated via `k.launch(stream,
  view)` does the right thing per `kind()`.
- **`KernelLoader`** — context handed to the loading lambda. `loader(f)`
  records a conservative `ReadWrite` token; refine via
  `loader.read / write / read_write / read_stencil`. `read_stencil`
  additionally snapshots a `HaloFactoryBase` so `Skeleton::compile` can
  auto-insert a halo update.
- **`ReductionResult<T>` + 3-stage reduce.** Stage 1: fused per-GPU
  `cub::BlockReduce` / `sycl::reduction`. Stage 2 (multi-GPU): peer-copy
  partials to the root, combine through `Op`. Stage 3 (distributed):
  `ncclAllReduce` (`reduce::op_kind<Op>` × `data_kind<T>` traits map
  to NCCL operator + dtype at compile time). Built-in op:
  `reduce::Sum<T>`. The OpenMP host reduce mirrors the device pipeline.

### Optional transports + I/O

- **MPI + NCCL** (`-DCARBON_ENABLE_MPI=ON`). `Distributed` pImpl over
  MPI lifetime, `detail::Communicator` over `ncclComm_t` (bootstrap via
  `ncclGetUniqueId` + `MPI_Bcast`), `DistTransfer<T>` for grouped
  Send/Recv, `dist_allreduce_in_place` for cross-rank reductions. MPI
  and NCCL are PRIVATE-linked; consumer TUs never see those headers.
  One-rank-per-GPU model under MPI today; hybrid multi-device-per-rank
  is the tracked follow-up.
- **HDF5** (`-DCARBON_ENABLE_HDF5=ON`). PRIVATE-linked; templates
  guarded by `CARBON_HAVE_HDF5`. With MPI it becomes `CARBON_HAVE_PHDF5`
  (collective MPI-IO writes into one file).
- **nanoVDB** (`-DCARBON_ENABLE_NANOVDB=ON`). Discovery: explicit
  include dir → `find_package(nanovdb)` → `FetchContent` OpenVDB
  `v11.0.0` (populate-only, skips TBB/Boost/blosc). Multi-rank writes
  one file per rank. Float / double / int32 today.
- **VTK legacy** — always-on, pure stdlib. `STRUCTURED_POINTS` BINARY
  with one `SCALARS name dtype N` block per field; one file per rank
  with `.rank<N>` suffix. Strict reader (rejects XML / non-BINARY /
  non-unit spacing / mismatched DIMENSIONS).
- **`carbon::io::write_native / read_native`** — multi-field per-rank
  binary format with explicit field metadata; the default for round-trip
  tests. Templated on Grid and shipping for both `Dense` and `DenseDisg`
  Cartesian families: the on-disk format is family-agnostic (header +
  per-field meta + SoA owned-cell payload) so a Dense-written shard can
  be read into a DenseDisg field of matching `global_extent` and vice
  versa. Per-row addressing inside the field's host mirror is the only
  family-specific bit, dispatched through `detail::row_offset` overloads
  on `Dense::Span` and `DenseDisg::Span`. `Block` native binary is
  out-of-scope here — the brick-list layout needs its own format and
  follows in the next P1.ε pass.

### Orchestration

- **`OverlappedStencil`** — two-stream helper. Issues halo on
  `halo_stream` concurrent with `compute(Internal)` on `compute_stream`,
  then fences `compute(Boundary)` behind the halo. `step()` tracks the
  iter-0 contract (skip the halo-event wait on the first call);
  `reset()` re-arms.
- **`Skeleton`** — token-graph + static multi-stream scheduler. `add(k)`
  appends; `compile()` walks every kernel pair and emits a directed
  edge on conflicting tokens (`Read, Read` is the only no-conflict
  pair). v2 scope: greedy assignment across carbon `Stream`s
  (uncapped by default, see `Options::max_streams`) with cross-stream
  events for in-iteration deps. Stream 0 is `sys.default_stream()`, and
  every `run()` is bracketed by a start/join fork-join on it, so nothing
  leaks across a run boundary — including between two *different*
  skeletons on the same `System` (an alternating ping-pong pair needs no
  host drain). `Options::overlap_mode
  { None, Standard, Extended, TwoWayExtended }` opts in to the OCC
  rewrite — auto-halo insertion + Internal/Boundary split + halo-stream
  isolation. Single-stream fast path is preserved bit-for-bit. Catches
  "halo missing before stencil" at compile time. Examples:
  `examples/jacobi_unified.cu` (Dense / DenseDisg / Block from one
  source file, runtime grid + transport + overlap selection);
  `examples/checkpoint_dense.cu` and `examples/checkpoint_sphere_image.cu`
  (Dense checkpoint I/O — the latter adds a sphere SDF field, restart
  verification, downsampled slice, and PGM export).

### Cross-grid concepts (lock the API contract)

- **`DenseLikeSpanConcept` / `BlockLikeSpanConcept` / `PartitionConcept`**
  in `include/carbon/detail/grid_concepts.h` document the minimum
  duck-typed surface every grid family's `Span` and `Partition` must
  expose. The `Span` shape splits along the iteration model:
  `DenseLikeSpanConcept` (Dense / DenseDisg) covers `iter_extent /
  logic_extent / xy_slab_elements / global_location`;
  `BlockLikeSpanConcept` (Block) covers `block_count / n_internal /
  iter_block_count(view) / view_block_idx / set_and_validate(b, ix,
  iy, iz) / linear_offset / per_card_elements /
  linear_storage_elements / global_location`. `PartitionConcept`
  (`read / write / neighbour` with `.value` + `.valid`) is shared
  by every family. `DeviceKernelImpl`'s ctor anchors **both** the
  matching `Span` concept @b and `PartitionConcept` via
  `static_assert` on each branch of `grid_has_dense_iteration_v<Grid>`
  (P1.β closure of `docs/audit_2026_05_third_pass.md`); a fourth
  grid family that ships a mis-shaped `Span` or `Partition` lands
  on a clean concept diagnostic at kernel-impl ctor time instead of
  a deep launch-path instantiation stack. `SpanConcept` is retained
  as a deprecated alias for `DenseLikeSpanConcept` for back-compat
  with the original concept name.

### Test count snapshot

Counts drift with every feature, so treat these as a sanity check, not a
contract — `ctest --test-dir <build> -N` is the source of truth. Measured
2026-07-31 on a 2×RTX 6000 Ada box (`sm_89`), every configuration fully green:

| Configuration | Tests |
| --- | --- |
| default (CUDA) | 93 |
| `-DCARBON_ENABLE_MPI=ON` (CUDA) | 116 † |
| `-DCARBON_BACKEND=Sycl` | 89 |
| `hip-nvidia` (HIP backend on NVIDIA) | 89 |

† not re-measured on 2026-07-31; carried over from the 2026-07-30 2×A100 run.

`hdf5_io` / `nanovdb_io` are registered unconditionally but compile-time
skip when their flags are off. The 26-test MPI delta is the `distributed`
ctest label and registers on both backends. The 4-test CUDA/SYCL delta is
the `carbon_add_cuda_only_unit_test` gate (`block_cuda` + the three
`mgrid_*` tests with hand-rolled `<<<>>>` kernels).
`carbon_format_check` is enforced on every backend.

### Packaging

`cmake --install` lays down a `find_package(carbon)` config under any
prefix: `lib/cmake/carbon/{carbonConfig,carbonConfigVersion,
carbonTargets}.cmake`, public headers under `include/carbon/`, bundled
rapidjson and spdlog. `carbonConfig.cmake` calls `find_dependency` for
OpenMP / CUDAToolkit / spdlog; MPI and NCCL stay PRIVATE-linked.

## Code style (enforced by `.clang-format`)

- **Signature layout — "divided" style.** Return type and qualifiers on one line; function name
  starts the next line. Multi-parameter signatures stack one parameter per line under the first.
  Applies even to two-parameter functions. `BinPackParameters: AlwaysOnePerLine`,
  `BreakAfterReturnType: All`.
- **Naming.** `carbon::` namespace. Types: PascalCase. Free functions & members: snake_case.
  Enum enumerators: PascalCase. Enum string forms: lowercase.
- **Integers.** Prefer bare `int64_t` over `std::int64_t` (aligns with CUDA idioms). A small
  number of legacy spots in `src/dense.cpp` still use `std::int64_t` / `std::size_t` and are
  scheduled for a normalization pass.
- **Constructors.** Member-initializer lists always stacked (`PackConstructorInitializers: Never`).
- **Access modifiers.** Flush with the class body (`AccessModifierOffset: -2`).
- **Comments.** Never auto-reflowed (`ReflowComments: Never`). Do not add narration comments that
  restate the code; only encode non-obvious intent.
- **Macros.** `CARBON_HD` = `__host__ __device__`; `CARBON_API` = export/import; `CARBON_LOG_*`
  are logging statements. All are registered in `.clang-format` so they format correctly.
- **Known benign quirks:** `AlwaysOnePerLine` can awkwardly wrap multi-argument template types and
  braced-init-lists inside `static_assert`. Workaround: add a local `using array_type = ...;`
  alias inside the struct. Documented in `.clang-format`.

## Conventions for agents

- Read-only files: everything under `Neon/`.
- After a substantive edit, always run the build + tests + `format-check` triad above. Do not
  announce completion until all three are clean.
- When porting a Neon feature, first read its header under `Neon/libNeon*/include/...`, then the
  matching `src/...` implementation, then mirror the semantics in the `carbon` namespace with the
  style above — do not copy Neon code verbatim; this is a clean-room rewrite.
- License for the rewrite: Apache-2.0 (same as Neon). See `LICENSE`.

## Project memory

Long-form design rationale lives in **[`docs/CHANGELOG.md`](docs/CHANGELOG.md)**
— append-only entries dated `YYYY-MM` covering every non-trivial design call.
When making a substantive change that future contributors will need context for,
add a dated bullet to that file rather than this one. Keep `AGENTS.md` focused
on *current* operational state.
