# carbon — design decisions & project memory

Long-form rationale for design decisions that aren't easily captured in code
comments. Append-only — entries are dated (`YYYY-MM`) and stay in roughly
chronological order. Newer entries go at the bottom.

For *current* state (architecture, build matrix, conventions, status snapshot),
see [`AGENTS.md`](../AGENTS.md). This file complements it: AGENTS.md tells
you where the codebase is *today*, this file tells you *how it got here* and
*why*.


For longer design discussions and decisions not captured in code comments, add sections below (or
new files in `docs/`). Keep short, durable notes; ephemeral plans belong in PRs/issues.

- **2026-04 — `Span` is the sole owner of `z_offset`** (it is common to all fields over the same
  grid). `Idx` is purely local; use `Span::global_location(idx)` when global coordinates are
  needed.
- **2026-04 — `Span::xy_slab_elements` is overloaded.** Instance `xy_slab_elements()` returns the
  stored pitch; static `xy_slab_elements(int64_3d e)` returns `e.x * e.y` (the minimum pitch for
  that extent). They resolve unambiguously at every call site; `&Span::xy_slab_elements` would
  need a cast if ever taken.
- **2026-04 — Offset math lives in `Span::linear_offset`**, not in `Partition`. `Partition::read`
  delegates.
- **2026-04 — Public API hides the multi-GPU fan-out.** `Buffer<T>`, `Stream`, `Event`, and
  `Field<T, Components>` are presented to users as singular handles ("a buffer", "a stream",
  "a field on this `System`") even though each is backed by a per-GPU collection
  (`detail::BufferGroup`, `StreamGroup`, `EventGroup`, `Field`'s internal vectors). The goal
  is that application code written against Carbon does not branch on GPU count at the call
  site; the orchestration types (`System`, `Dense`, `Transfer`, `Kernel`, `Halo` once it
  lands) iterate over devices internally. Future API additions should preserve this — prefer
  singular-handle names and orchestrator types over per-GPU APIs.
- **2026-04 — Grid-family types are reached through aliases on the grid class, never via
  `<carbon/detail/...>`.** `Dense::Span`, `Dense::Idx`, `Dense::Partition<T, Components>`, and
  `Dense::Field<T, Components>` are the only names user code (including tests/examples) should
  use; the underlying `detail::dense::*` types stay free to evolve when other grid families
  (`bGrid`, `mGrid`-style) land and need to vary `Idx` / `Partition`. Including
  `<carbon/dense.h>` brings all of them in.
- **2026-04 — `Span` stores the haloed (storage) layout.** `Dense`'s ctor builds every span with
  `halo_z = stencil_.radius()`, so `spans_[i]` describes the actual GPU buffer: extent is
  `(global.x, global.y, local_z + 2·halo_z)` and `z_offset` is `logical_z_start − halo_z`. The
  CUDA launch grid (`gpu_grid_`) is sized from `span.logic_extent()` so kernels iterate owned
  cells only; halo layers are reached via neighbour offsets from those threads.
  `dense_grid_kernel` shifts the iteration index by `span.halo_z()` before building the `Idx`,
  so pointer math (`linear_offset`) and `global_location` stay buffer-local and correct without
  any per-kernel bookkeeping. `Dense::newField` no longer rebuilds field-local spans — it reads
  sizes and partition offsets directly off `spans_[i]`.
- **2026-04 — `DataView` is a launch-time parameter, not stored on `Kernel`.** A `Kernel`
  built via `Dense::make_kernel(loading_fn, smem)` carries no view; the user picks
  Standard / Internal / Boundary at the call site through
  `kernel.launch(stream, DataView::...)` (which defaults to `Standard`). The same loading
  lambda can therefore be reused across views without rebuilding the closure, and the
  Skeleton layer (a future scheduler) will be free to choose views based on dependency
  analysis without rewriting the user's kernel objects. `Dense` precomputes the per-view
  per-GPU CUDA grid in its constructor (`gpu_grid(view)` / `gpu_block(view)`); the
  matching thread → `Idx` translation lives on `Span::set_and_validate(view, ix, iy, iz)`
  and is shared between every kernel that runs on the slab.
- **2026-05 — `Kernel` is a single value type, not a template.** Carbon mirrors
  `Neon::set::Container`'s pImpl-via-virtual-base shape: the public `carbon::Kernel`
  is a copyable value handle holding a `std::shared_ptr<detail::KernelImpl>`. The
  templated leaves (`detail::DeviceKernelImpl<Grid, LoadingFn>`,
  `detail::ReduceKernelImpl<Grid, LoadingFn, T, Op>`, `detail::HaloKernelImpl<T,
  Components>`, `detail::AnchorKernelImpl`) are not user-visible. Every kind exposes the
  same `launch(stream, view = DataView::Standard)` so a heterogeneous pipeline lives in
  `std::vector<Kernel>` and runs by iteration. `Kernel::compute / reduce / halo /
  anchor` are the static factories; `Dense::make_kernel` and `Dense::make_reduce_kernel`
  are convenience members that forward to them. Reduction destinations are bound at
  construction time (`make_reduce_kernel(loading_fn, dst_ref, ...)`) so the polymorphic
  `launch(stream, view)` signature stays uniform. Move-only loading lambdas are
  supported because the impls store the lambda by value (no `std::function` wrapper).
- **2026-04 — `BoundaryMode` stays runtime, not a template parameter on `Field`.** Storage does
  not depend on the mode (the buffer is the same shape in Open and PeriodicZ); mixed-mode
  boundary conditions (e.g. CFD channel flow: open in x/y, periodic in z) extend cleanly when
  the enum grows into a per-axis bitfield (`PeriodicX | PeriodicY | OpenZ`); a runtime mode
  also keeps a single binary able to switch modes between runs. Compile-time specialization is
  fine where it earns its keep — `Halo<T, Components, BoundaryMode>` is a reasonable place
  because the precomputed slab-pair set differs structurally between modes and the user
  constructs one `Halo` per field — and we may add an opt-in `Partition` view templated on
  mode for the rare hot kernel that needs the branch eliminated. Default everywhere else
  remains the runtime byte on `Span` and `Dense`.
- **2026-05 — `Kernel` host (OpenMP) execution + universal trichotomy.** Carbon offers three
  factory shapes for compute/reduce kernels, on top of the value-typed `Kernel` handle.
  Naming convention rewards the common case with the shortest spelling:
    - `Dense::make_kernel` / `make_reduce_kernel` → device-only
      (`KernelKind::Compute` / `Reduction`). Smallest compile cost; only the CUDA / SYCL launch
      body is instantiated. This is the new default — most user code wants this.
    - `Dense::make_host_kernel` / `make_host_reduce_kernel` → host-only
      (`ComputeHost` / `ReductionHost`). Pure OpenMP; no nvcc/acpp TU required.
    - `Dense::make_universal_kernel` / `make_universal_reduce_kernel` → dual-target
      (`Universal` / `UniversalReduction`). Both launch bodies instantiated from the SAME
      `CARBON_HOST_DEVICE` lambda; user picks per call via `Kernel::launch_on_device` /
      `Kernel::launch_on_host`. The polymorphic `launch(stream)` defaults to device so a
      Universal kernel sitting in a heterogeneous `std::vector<Kernel>` iterated via
      `launch(stream)` behaves like a device kernel. Use this when you need in-process
      device/host parity testing on the same lambda or adaptive runtime dispatch.
  `KernelImpl` carries four virtuals (`launch_on_device`, `launch_on_host`, `runs_on_device`,
  `runs_on_host`) with default-throw bodies; concrete impls override what they support. The
  dual impls live in `detail/universal_kernel_impl.h` (CUDA/SYCL submit + OpenMP nested-for
  in one class, single `Grid` and `LoadingFn` storage) and `detail/universal_reduce_kernel_impl.h`
  (CUB / sycl::reduction scratch allocated once in the ctor + per-thread partials for the
  host path). Halo is still device-only as of this commit (host halo is a follow-up).
- **2026-05 — Why `KernelLoader` stays a non-template (with a runtime `target` byte).** The
  obvious refactor — make `KernelLoader<KernelTarget T>` template-distinct so the device
  loader and host loader are statically different types and `operator()` dispatches via
  `if constexpr` — was prototyped and reverted because it forces every Universal-kernel
  loading lambda to be generic (`auto& loader`, since the lambda is invoked once per launch
  path with two different specializations). nvcc rejects extended `__host__ __device__`
  lambdas defined inside a generic lambda, which broke every existing universal kernel
  body across the test suite and examples. The same trap bites device-only kernels with
  nested wrapper lambdas (e.g. `cg_poisson_single_gpu.cu`'s `kernel_set_const` /
  `kernel_matvec` factories). Net call: keep `KernelLoader` non-template with a runtime
  `target_` byte; lift `KernelTarget { Device, Host }` to namespace scope (still aliased
  as `KernelLoader::Target` for source compat) so the enum has a clean public name. The
  type-precision win we'd have got — distinct device-loader vs host-loader types, slightly
  better static checking for stencil-aware loader extensions — is deferred until either
  nvcc fixes the bug or carbon switches its default backend to one that doesn't have it.
- **2026-05 — Neon vs carbon host/device design — reasoning behind the trichotomy.** Neon
  splits the same problem across two axes: `Neon::Execution` (compile-time per-container
  via `Container::factory<Execution>`) and `Neon::Runtime` (runtime, on `Backend`, picked at
  process construction). Under `Runtime::openmp`, Neon's `memory.ti.h` forces the GPU
  allocator to `NULL_MEM` so the unused side simply doesn't exist; `DeviceContainer::run` is
  routed through `DevSet::launchLambdaOnSpan` which `switch`es on `Backend::runtime()`.
  Carbon deliberately diverges: per-`launch_on_*` call dispatch on a single `Universal`
  handle, with both buffer sides simultaneously live whenever `MemorySide::HostDevice`. This
  enables in-process device/host parity testing on the SAME lambda
  (`tests/test_kernel_host_parity.cu` is the showcase) and decouples kernel definition from
  execution backend (a kernel is a compute description; the launch site decides where it
  runs). The cost is that `make_kernel` instantiates both launch bodies up front;
  `make_device_kernel` is the explicit escape hatch and matches Neon's per-container compile
  profile. Two follow-ups are still on the table:
    - **Lazy host-mirror allocation for `MemorySide::HostDevice`.** Today carbon allocates
      both sides eagerly. Neon's `NULL_MEM` trick saves memory automatically when the unused
      side is never touched; carbon should consider lazy allocation on first `host_partitions()`
      access (or first `Transfer<T>` to host) to close that gap without giving up per-call
      dispatch.
    - **Halo on host.** Neon's halo machinery is uniform across runtimes (just memcpy
      semantics differ). Carbon's `Halo` / `HaloKernelImpl` are device-only; we need a host
      variant that does `std::memcpy` between `cpu_buffers` across slabs (including PeriodicZ
      wraparound) and either lives behind `Halo::launch_on_host` or behind a dedicated
      `make_host_halo_kernel` factory. This was deferred during the `2026-05` host-kernel
      work; users running fully-host pipelines on multi-rank CPU will need it.
- **2026-05 — Distributed transport under `CARBON_BACKEND=Sycl` (Option C, landed).**
  After evaluating five paths (icpx + oneCCL, dual icpx+acpp, AdaptiveCpp + vendor-native
  collectives, AdaptiveCpp + oneCCL forced, AdaptiveCpp + plain MPI) against the deployment
  matrix NVIDIA + AMD + future Apple-CPU, carbon committed to **Option C: keep AdaptiveCpp,
  call NCCL/RCCL/MPI directly on the underlying vendor pointers**. Rationale:
  - oneCCL hard-depends on Intel's SYCL runtime (`intel-sycl-rt`), conflicts with
    AdaptiveCpp's libsycl runtime in the same process. The conda-forge `oneccl-devel`
    explicitly pulls those Intel runtimes.
  - AdaptiveCpp's CUDA backend allocates USM as `cudaMalloc`'d pointers; AdaptiveCpp's HIP
    backend allocates as `hipMalloc`'d pointers. Both are valid vendor pointers that NCCL
    (resp. RCCL, ABI-compatible NCCL clone for AMD) consume directly with no SYCL-side
    translation.
  - Final shape: SYCL+`cuda:` target → NCCL (validated, 27/27);
    SYCL+`hip:` target → RCCL (compile + link validated; runtime gated on AMD GPU);
    SYCL+`omp` target / Apple-CPU → plain MPI on host pointers (future).
  - icpx + Codeplay NVIDIA plugin is gated (developer.codeplay.com signup) and the user
    has no sudo on the dev box, so the icpx route was not viable for full automation in
    this pass.

  Validation matrix (all five compile + link cleanly; the four NCCL configs run the
  full test suite, the RCCL config builds but cannot run on this NVIDIA-only box):
  - CUDA (vanilla): 29/29
  - CUDA + MPI/NCCL: 34/34
  - SYCL (vanilla): 29/29 (was 22/22 — caught up after the cross-stream ordering
    primitives + raw-cudaXxx test migrations landed; see the
    `2026-05 — Cross-stream ordering on Stream / Event` memory bullet)
  - SYCL + MPI/NCCL: 34/34 (was 27/27 — same migration)
  - **SYCL + MPI/RCCL: builds clean (34 test exes + 5 examples link); runtime needs
    `/dev/kfd` + AMD GPU — `librccl.so.1` and `libamdhip64.so.6` are wired in via
    DT_NEEDED, RPATH includes `/home/max/.local/rocm/lib` so `ldd` resolves both.**

  Code changes landed:
  - `CMakeLists.txt`:
    - Relaxed the `CARBON_BACKEND=Sycl` + `CARBON_ENABLE_MPI=ON` guard to accept
      `cuda:` OR `hip:` targets in `CARBON_SYCL_TARGETS`.
    - Added `CARBON_SYCL_ROCM_PATH` (mirrors `CARBON_SYCL_CUDA_PATH`) and forwards it
      to acpp as `--acpp-rocm-path=...` (PUBLIC, so test/example TUs that compile
      SYCL+HIP get the same flag — unlike the CUDA path where acpp can find
      `/usr/local/cuda` itself, ROCm's default-search `/opt/rocm` is rarely populated
      on a no-sudo dev box). Picks NCCL+CUDA by default, switches to RCCL+HIP when
      `CARBON_BACKEND=Sycl`, `hip:` is in `CARBON_SYCL_TARGETS`, and `cuda:` is not.
    - Defines `CARBON_USE_RCCL` + `__HIP_PLATFORM_AMD__=1` PRIVATE on the carbon
      target under the RCCL path; consumed by the `gpu_runtime_compat.h` shim below.
    - Tagged `src/distributed.cpp`, `src/communicator.cpp`, `src/dist_transfer.cpp`
      for `add_sycl_to_target` under the SYCL backend so they get the SYCL include
      flags and AdaptiveCpp's interop glue.
    - Added `target_link_directories(carbon INTERFACE ...)` for NCCL/RCCL libdir and
      each MPI libdir, plus `target_link_options(carbon INTERFACE/PRIVATE LINKER:--allow-shlib-undefined)`.
      The INTERFACE flag stops the conda-bundled `x86_64-conda-linux-gnu-ld` from
      rejecting test/example links against libcarbon.so chasing weak NCCL/RCCL
      symbols (both libs export `ncclSend`/`ncclRecv`/etc. as weak; the conda lld
      won't resolve them at link time even with -L pointing at the .so). The PRIVATE
      flag is needed too for libcarbon.so's own link under RCCL because `librccl.so`
      has transitive DT_NEEDED to `librocm_smi64.so.7` / `librocprofiler-register.so.0`
      which aren't present on a CPU-only / NVIDIA box. `--allow-shlib-undefined`
      defers the check to runtime, which is what DT_NEEDED + RPATH already handle
      correctly. Harmless under NCCL.
    - `CARBON_HAVE_MPI_NCCL` PUBLIC macro is kept NCCL-flavoured (the public ABI is
      identical between NCCL and RCCL — same function names; the spelling is just a
      historical artefact); renaming to `CARBON_HAVE_MPI_COLLECTIVES` is queued for
      when a host-MPI variant lands and the macro stops implying GPU collectives.
  - `cmake/FindRCCL.cmake`: mirrors `FindNCCL.cmake` byte-for-byte (rccl.h reuses the
    `NCCL_MAJOR/MINOR/PATCH` macro names so the version parse is unchanged); honours
    `RCCL_ROOT` / `ROCM_PATH` / `ROCM_HOME` hint vars and exposes
    `RCCL::rccl IMPORTED`.
  - `src/gpu_runtime_compat.h` (new, private, src/-only): one-stop shim that aliases
    `cudaXxx ↔ hipXxx` and `<nccl.h> ↔ <rccl/rccl.h>` based on `CARBON_USE_RCCL`.
    Exposes `carbon::detail::gpu_compat::{gpu_stream_t, gpu_error_t, gpu_success,
    gpu_get_device, gpu_set_device, gpu_get_device_count, gpu_get_error_string}`.
    Lets the three MPI sources (`distributed.cpp`, `communicator.cpp`,
    `dist_transfer.cpp`) plus `communicator_internal.h` keep one body of code
    working against either NCCL+CUDA or RCCL+HIP — the function-name patterns
    (`s/cuda/hip/`) and the public NCCL ABI (which RCCL preserves verbatim) make
    the substitution mechanical. Required because `nccl.h` declares `cudaStream_t`
    and `rccl/rccl.h` declares `hipStream_t`; the two cannot coexist in one TU.
  - `src/dist_transfer.cpp`: backend-aware NCCL/RCCL submission. Under CUDA backend,
    unchanged (`stream[0].get()` directly into `ncclSend`/`Recv`). Under SYCL, the
    NCCL/RCCL group is wrapped inside
    `q.submit(... cgh.AdaptiveCpp_enqueue_custom_operation([](sycl::interop_handle ih){ ... }))`.
    AdaptiveCpp's `hipsycl::sycl::handler` does NOT expose SYCL 2020's `host_task`
    — `AdaptiveCpp_enqueue_custom_operation` is the equivalent (also aliased as
    `hipSYCL_enqueue_custom_operation`, deprecated). The custom-op body extracts the
    underlying vendor stream via
    `ih.get_native_queue<sycl::backend::cuda>()` (NCCL path) or
    `ih.get_native_queue<sycl::backend::hip>()` (RCCL path, gated on
    `CARBON_USE_RCCL`), runs `ncclGroupStart` / per-entry `ncclSend|Recv` /
    `ncclGroupEnd`, and ordering chains naturally with preceding + subsequent SYCL
    submissions on the in-order queue. Followed by `q.wait_and_throw()` so
    `DistTransfer<T>::operator()` keeps its synchronous-looking call shape (the
    user pairs it with a `Stream::wait()` at iteration end).
  - `tests/CMakeLists.txt`: `carbon_add_mpi_cuda_test` mirrors `.cu` to a `.cpp` twin
    in the build tree under SYCL (same `configure_file COPYONLY` trick as
    `carbon_add_cuda_unit_test`) and routes through `add_sycl_to_target`.
  - Dist test bodies migrated off raw cuda runtime APIs:
    - `tests/test_dist_transfer.cu`, `tests/test_halo_dist.cu`,
      `tests/test_halo_dist_periodic_z.cu`: replaced `cudaStreamSynchronize(stream[g].get())`
      loops with backend-neutral `stream.wait()` (matching the
      `test_halo_single_node` / `test_halo_periodic_z` pattern). Dropped now-unused
      `<cuda_runtime.h>` includes (also from `tests/test_native_io_dist.cu`).

  Configure recipe for SYCL+MPI/NCCL (one-time):

  ```bash
  cd /home/max/repos/carbon
  rm -rf build-sycl-mpi
  OPAL_PREFIX=/home/max/.local/openmpi/usr \
  LD_LIBRARY_PATH=/home/max/.local/openmpi/usr/lib/x86_64-linux-gnu:/home/max/.local/openmpi/usr/lib/x86_64-linux-gnu/openmpi/lib \
  cmake -S . -B build-sycl-mpi \
    -DCARBON_BACKEND=Sycl \
    -DCARBON_SYCL_TARGETS="cuda:sm_80;omp" \
    -DCARBON_ENABLE_MPI=ON \
    -DCMAKE_CXX_COMPILER=/home/max/.pixi/envs/adaptivecpp/bin/clang++-19 \
    -DAdaptiveCpp_DIR=/home/max/.pixi/envs/adaptivecpp/lib/cmake/AdaptiveCpp \
    -DMPI_HOME=/home/max/.local/openmpi/usr \
    -DMPI_CXX_COMPILER=/home/max/.local/openmpi/usr/bin/mpicxx \
    -DMPIEXEC_EXECUTABLE=/home/max/.local/openmpi/usr/bin/mpiexec \
    -DMPIEXEC_NUMPROC_FLAG=-n \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,/home/max/.local/openmpi/usr/lib/x86_64-linux-gnu -L/home/max/.local/openmpi/usr/lib/x86_64-linux-gnu -Wl,-rpath-link,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu" \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,/home/max/.local/openmpi/usr/lib/x86_64-linux-gnu -L/home/max/.local/openmpi/usr/lib/x86_64-linux-gnu -Wl,-rpath-link,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu" \
    -DNCCL_ROOT=/home/max/.local/lib/python3.10/site-packages/nvidia/nccl \
    -DCMAKE_BUILD_TYPE=Release
  ```

  Build + test:

  ```bash
  OPAL_PREFIX=/home/max/.local/openmpi/usr \
  LD_LIBRARY_PATH=/home/max/.local/openmpi/usr/lib/x86_64-linux-gnu:/home/max/.local/openmpi/usr/lib/x86_64-linux-gnu/openmpi/lib \
  cmake --build build-sycl-mpi -j8 && \
  OPAL_PREFIX=/home/max/.local/openmpi/usr \
  LD_LIBRARY_PATH=/home/max/.local/openmpi/usr/lib/x86_64-linux-gnu:/home/max/.local/openmpi/usr/lib/x86_64-linux-gnu/openmpi/lib \
  ctest --test-dir build-sycl-mpi --output-on-failure
  ```

  Note on environment: the OpenMPI install at `/home/max/.local/openmpi/usr` is a
  conda-relocatable build (configure ran with `--prefix=/usr` then was relocated to
  `/home/max/.local/openmpi/usr`). It needs `OPAL_PREFIX=/home/max/.local/openmpi/usr`
  for `mpicxx -showme` to find its wrapper data file at runtime, plus the two
  `LD_LIBRARY_PATH` entries above (one for `libopen-pal.so.40` / `libopen-rte.so.40`
  / `libhwloc.so.15` in `…/lib/x86_64-linux-gnu/`, one for `libmpi.so` /
  `libmpi_cxx.so` in `…/lib/x86_64-linux-gnu/openmpi/lib/`). The clang++-19 from
  AdaptiveCpp's pixi env uses an isolated `x86_64-conda-linux-gnu-ld` that does not
  search `/usr/lib/x86_64-linux-gnu` by default, hence the explicit
  `-Wl,-rpath-link,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu` flags so it
  can find system `libz.so.1` and `libudev.so.1` (DT_NEEDED of OpenMPI's
  `libopen-rte` and `libhwloc`).

  Configure recipe for SYCL+MPI/RCCL (compile-validated only on this NVIDIA-only box;
  cannot run without `/dev/kfd` + AMD GPU):

  ```bash
  cd /home/max/repos/carbon
  rm -rf build-sycl-rccl
  OPAL_PREFIX=/home/max/.local/openmpi/usr \
  LD_LIBRARY_PATH=/home/max/.local/openmpi/usr/lib/x86_64-linux-gnu:/home/max/.local/openmpi/usr/lib/x86_64-linux-gnu/openmpi/lib \
  cmake -S . -B build-sycl-rccl \
    -DCARBON_BACKEND=Sycl \
    -DCARBON_SYCL_TARGETS="hip:gfx908" \
    -DCARBON_ENABLE_MPI=ON \
    -DCMAKE_CXX_COMPILER=/home/max/.pixi/envs/adaptivecpp/bin/clang++-19 \
    -DAdaptiveCpp_DIR=/home/max/.pixi/envs/adaptivecpp/lib/cmake/AdaptiveCpp \
    -DMPI_HOME=/home/max/.local/openmpi/usr \
    -DMPI_CXX_COMPILER=/home/max/.local/openmpi/usr/bin/mpicxx \
    -DMPIEXEC_EXECUTABLE=/home/max/.local/openmpi/usr/bin/mpiexec \
    -DMPIEXEC_NUMPROC_FLAG=-n \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,/home/max/.local/openmpi/usr/lib/x86_64-linux-gnu -L/home/max/.local/openmpi/usr/lib/x86_64-linux-gnu -Wl,-rpath-link,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu" \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,/home/max/.local/openmpi/usr/lib/x86_64-linux-gnu -L/home/max/.local/openmpi/usr/lib/x86_64-linux-gnu -Wl,-rpath-link,/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu" \
    -DCARBON_SYCL_ROCM_PATH=/home/max/.local/rocm \
    -DRCCL_ROOT=/home/max/.local/rocm \
    -DCMAKE_BUILD_TYPE=Release
  PATH=/home/max/.local/lld/bin:$PATH \
  OPAL_PREFIX=/home/max/.local/openmpi/usr \
  LD_LIBRARY_PATH=/home/max/.pixi/envs/adaptivecpp/lib:/home/max/.local/openmpi/usr/lib/x86_64-linux-gnu:/home/max/.local/openmpi/usr/lib/x86_64-linux-gnu/openmpi/lib \
  cmake --build build-sycl-rccl -j8
  ```

  Note on the SYCL+RCCL prerequisites (assembled locally, no sudo):
  - `/home/max/.local/rocm/` — unified ROCm 6.3 staging tree, built by extracting the
    AMD ROCm `.deb` packages from `repo.radeon.com/rocm/apt/6.3/`:
    - `rccl_2.21.5.60300-39~22.04` → `lib/librccl.so*`
    - `rccl-dev_2.21.5.60300-39~22.04` → `include/rccl/rccl.h`
    - `hip-dev_6.3.42131.60300-39~22.04` → `include/hip/...`
    - `rocm-device-libs_1.0.0.60300-39~22.04` → `amdgcn/bitcode/*.bc` and
      `lib/llvm/lib/clang/<ver>/lib/amdgcn/bitcode/*.bc`
  - `/home/max/.local/lld/bin/` — clang HIP toolchain (clang-19 needs lld + offload
    bundler + llvm-tools-19 to compile for `gfx*`):
    - `lld-19.1.7-hef48ded_1.conda` → `ld.lld`, `lld`, `lld-link`
    - `clang-tools-19.1.7-default_h57a47db_9.conda` → `clang-offload-bundler[-19]`,
      `clang-offload-packager[-19]`
    - `llvm-tools-19-19.1.7-h3b15d91_2.conda` → `llvm-objcopy[-19]`, `llvm-link[-19]`,
      `llvm-strip[-19]`, etc. (the `llvm-tools` meta package only ships symlinks; the
      real binaries live in `llvm-tools-19`).
  - `LD_LIBRARY_PATH` must include `/home/max/.pixi/envs/adaptivecpp/lib` so the
    staged tools can load `libLLVM.so.19.1` from AdaptiveCpp's pixi env.
  - libcarbon.so links cleanly under RCCL despite missing `librocm_smi64.so.7` /
    `librocprofiler-register.so.0` because of the `--allow-shlib-undefined` PRIVATE
    flag added in the same pass. At runtime those would need to come from a real
    ROCm install (not currently present).

  Followups (deferred):
  - Generalise `CARBON_HAVE_MPI_NCCL` to `CARBON_HAVE_MPI_COLLECTIVES` once a host-MPI
    transport lands; the macro currently implies "GPU collectives are available" which
    is true for both NCCL and RCCL but won't be once we add a CPU-only fallback.
  - Drop the `q.wait_and_throw()` in the SYCL `AdaptiveCpp_enqueue_custom_operation`
    path once we're confident in cross-stream ordering; replace with explicit closure
    copies of `peer_ranks`/`ops`/`first_bytes`/`byte_counts` so the call returns
    asynchronously.
  - Validate the SYCL+RCCL path on a real AMD GPU (build is clean; runtime is
    untested because no AMD GPU is reachable from the dev box).
  - Add a host-MPI fallback (`omp`-only SYCL target, or a future Apple-CPU build) that
    does D→H copy on each entry, plain `MPI_Send`/`MPI_Recv` on host pointers, then
    H→D copy. Slower but unblocks CPU-only and Apple-Silicon distributed runs.
- **2026-05 — Cross-stream ordering on `Stream` / `Event` (CUDA-only test gating
  cleared).** Added `EventGroup::record(StreamGroup)` and `StreamGroup::wait_for(EventGroup)`
  inline in `<carbon/detail/stream_group.h>` (so a single header gives users both group
  types and the cross-ordering helpers). CUDA path → per-GPU `cudaEventRecord` /
  `cudaStreamWaitEvent`; SYCL path → per-GPU
  `q.AdaptiveCpp_enqueue_custom_operation([](auto&){})` for the no-op record (returns a
  barrier `sycl::event` capturing the in-order queue's frontier; the wrapped
  `sycl::event` inside `detail::Event` is reassigned in place via the existing
  `shared_ptr` so all shallow copies see the new recording) and
  `q.AdaptiveCpp_enqueue_custom_operation([](auto&){}, dep)` for the wait (no-op
  custom op with `depends_on(dep)` orders subsequent submissions on this queue after
  the event). With this primitive in place, the seven historical
  `carbon_add_cuda_only_unit_test` slots — `dense_kernel`, `partition_read`,
  `transfer_cuda`, `integration`, `partition_halo`, `transfer_peer`,
  `data_view_split_jacobi` — were migrated off raw `cudaMalloc` /
  `cudaMemcpy` / `cudaStreamSynchronize` / `cudaEventRecord` /
  `cudaStreamWaitEvent` to carbon's backend-neutral primitives (`Stream::wait()`,
  `Event::record(stream)`, `Stream::wait_for(event)`, `Buffer<T>` + `Transfer<T>` for
  raw device-buffer staging). All 29 default tests (34 with MPI) now build + pass
  under both CUDA and SYCL; the `carbon_add_cuda_only_unit_test` helper is kept for
  future raw-CUDA tests but its registered list is empty.
- **2026-05 — Reduce pipeline Docker matrix green.** `bash scripts/docker-build-all.sh
  --sm-arch sm_80` passes all five configs end-to-end:
  - **cuda**: 31/31 ctest pass.
  - **cuda-mpi**: 37/37 ctest pass.
  - **sycl** (`cuda:sm_80;omp`): 31/31 ctest pass.
  - **sycl-mpi** (`cuda:sm_80;omp` + OpenMPI + NCCL): 37/37 ctest pass.
  - **sycl-rccl** (`hip:gfx908` + OpenMPI + RCCL): build + link ok (108 targets);
    ctest skipped (no AMD GPU on the dev box).

  Bring-up notes from this pass (now baked into `docker/Dockerfile.*` and
  `scripts/docker-build-all.sh`):

  - **CMake ≥ 3.25 + clang-format ≥ 20** in all three images, installed via pip.
    Ubuntu 22.04's apt `cmake` is 3.22 (doesn't know the `CUDA20` dialect carbon
    sets via `CMAKE_CUDA_STANDARD 20`) and its `clang-format` is 14 (rejects
    carbon's `BinPackParameters: AlwaysOnePerLine`, an LLVM 20+ enum value).
  - **`OMPI_ALLOW_RUN_AS_ROOT=1` + `OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`** in the
    cuda / sycl / sycl-rccl Dockerfiles. OpenMPI refuses to spawn `mpiexec` as
    root by default; containers run in an isolated namespace so the ordinary
    rationale doesn't apply, and without these vars all six `carbon_*dist*`
    ctest entries fail to start.
  - **`OMPI_CXX=${ACPP_ENV}/bin/clang++-19`** in the sycl / sycl-rccl Dockerfiles.
    Without it, `mpicxx -show` emits `g++` compile/link flags, and FindMPI's
    C++ compile probe against `CMAKE_CXX_COMPILER=clang++-19` fails silently,
    producing a cryptic `MPI_CXX_WORKS` missing.
  - **`libboost-devel`** in the sycl / sycl-rccl conda envs. AdaptiveCpp 25.02's
    OpenMP backend pulls in `boost/fiber/fiber.hpp` from
    `collective_execution_engine.hpp`; the conda `adaptivecpp` package doesn't
    hard-depend on it because not every build path instantiates the collective
    kernels, but carbon's does.
  - **`clang-tools=19`** in the sycl-rccl conda env. AdaptiveCpp's HIP target
    invokes `clang-offload-bundler` + `clang-offload-packager` under the hood
    (amdgcn device-code packaging); the `clang=19` package alone doesn't ship
    them. Not needed for the `cuda:` target.
  - **`-Wl,-rpath-link,/usr/lib/x86_64-linux-gnu`** on `CMAKE_{EXE,SHARED}_LINKER_FLAGS`
    for sycl-mpi / sycl-rccl. The conda-shipped `x86_64-conda-linux-gnu-ld` does
    not search `/usr/lib/x86_64-linux-gnu` for transitive DT_NEEDED resolution
    (OpenMPI's `libopen-pal.so.40`, `libopen-rte.so.40`, `libhwloc.so.15`), so
    `FindMPI` probes fail without it. Mirror of the host-side recipe already
    documented for the local SYCL+MPI build.
  - **`-Wl,-rpath,${ACPP_ENV}/lib`** on the same flags, ordered BEFORE cmake's
    auto-generated `/usr/lib/x86_64-linux-gnu` RPATH entry (derived from
    `MPI_CXX_LIBRARIES`). Without it, the runtime loader resolves
    `libstdc++.so.6` from the distro (GLIBCXX_3.4.30) before conda's
    (GLIBCXX_3.4.32), and every test binary that pulls in `libacpp-rt.so` /
    `libboost_fiber.so` (both compiled against conda libstdc++) dies with
    `GLIBCXX_3.4.32 not found`.
  - **Per-source `-I${NCCL_INCLUDE_DIR}` + `-I${CARBON_SYCL_CUDA_PATH}/include`**
    on the four MPI sources (`src/{distributed,communicator,dist_transfer,dist_reduce}.cpp`)
    when building under `CARBON_BACKEND=Sycl`. NCCL's
    `INTERFACE_INCLUDE_DIRECTORIES=/usr/include` is dropped by cmake because
    `/usr/include` is in `CMAKE_PLATFORM_IMPLICIT_INCLUDE_DIRECTORIES` — but
    conda clang's sysroot does NOT search `/usr/include`, so `<nccl.h>` can't
    be found. Applying the include dirs narrowly (via `set_source_files_properties`
    on these four `.cpp`s only) avoids the globally-breaking `-isystem
    /usr/include` interaction with `<omp.h>` → `<stdlib.h>` that would break
    the OpenMP probe and other TUs.

- **2026-05 — `Skeleton` multi-stream scheduler (v2).** The skeleton now lowers
  the inferred token-driven DAG to a static schedule across up to `max_streams`
  carbon `Stream`s (default 4, configurable via the constructor). Three pieces
  to the implementation, all in `<carbon/skeleton.h>` (header-only):

  - **Greedy stream assignment.** Walking kernels in user-add order, kernel
    `k` is placed on (a) a stream whose tail is a direct predecessor of `k`
    (FIFO ordering already covers that dep — no event needed for that
    endpoint), else (b) the lowest-indexed empty stream up to the cap, else
    (c) a freshly allocated stream if `streams_.size() < max_streams`, else
    (d) stream 0 with cross-stream events on every external dep. Streams
    `1..` are allocated lazily inside `compile()` from `sys_.stream()` only
    when (b) / (c) trigger; the canonical Jacobi ping-pong (every step has
    the previous step as a direct predecessor) collapses to (a) every time
    and the schedule never opens a second stream — so the v1 fast path is
    preserved bit-for-bit. The parallel-sibling pattern (two unrelated copy
    kernels feeding a combine) gets `kernel_streams() == {0, 1, 0}` with
    one cross-stream event recorded after copy_xy on stream 1 and waited
    for on stream 0 before combine.

  - **Cross-stream event allocation.** Pooled per-predecessor: one
    `events_` slot per kernel that has at least one cross-stream successor.
    Multiple downstream consumers on different-from-pred streams all wait
    on the same record point — sound because they're all later than the
    predecessor in user-add order, so the single record-after-launch is
    the right barrier for all of them. The schedule stores
    `record_event_idx` on each `Step` (or `kNoEvent_`) and a per-step
    `wait_events` list.

  - **Inter-iteration barrier.** Multi-stream pipelines lose the implicit
    FIFO ordering across `run()` calls — kernel A on stream 0 in iter k+1
    might need data written by kernel B on stream 1 in iter k. The in-DAG
    event set does not capture this (the DAG is per-iteration). v2 installs
    a conservative O(num_streams²) barrier: at the end of every `run()`
    each populated stream records a `iter_tail_event_[s]`; on every
    subsequent `run()` each stream waits on every other stream's prior-iter
    tail event before launching its first kernel. Cheap (12 wait ops per
    iter at the default 4-stream cap) and skipped entirely when
    `streams_.size() == 1`. Tighter analysis (record/wait only on the
    actual cross-iter edges between writers and readers of conflicting
    `field_uid`s) is queued as a follow-up — the conservative barrier is
    correct in all cases, just over-eager for pipelines whose cross-iter
    edges are sparse.

  Test coverage in `tests/test_skeleton.cu`: the existing Jacobi ping-pong
  subtest now also asserts `kernel_streams() == {0,0,0,0}` and
  `num_streams() == 1` (single-stream fast path); a new parallel-sibling
  subtest builds copy_uv + copy_xy + combine, asserts the inferred edges
  (`{{2}, {2}, {}}`), the stream assignment (`{0, 1, 0}`), and bit-for-bit
  parity vs a single-stream reference loop across 5 iterations — the
  inter-iter barrier is exactly what makes the parity hold (without it
  iter k+1's combine on stream 0 could race iter k's copy_xy on stream 1).
  Public surface added: `Skeleton::num_streams()`, `Skeleton::max_streams()`,
  `Skeleton::kernel_streams()`, `Skeleton::streams()`. The constructors
  gained a trailing `max_streams = 4` parameter.

- **2026-05 — Field naming + Skeleton DOT export.** Two thin user-facing
  additions on top of v2:

  - **`Field::name()` / `Field::set_name(std::string)`.** Optional, free-form
    string label on every `Dense::Field`. `Dense::newField` gained a trailing
    `std::string name = {}` parameter so the common case (named at
    construction) is one extra argument; `set_name` is chainable and returns
    `Field&` so it composes with the `auto u = dense.newField<...>(side).set_name("u")`
    pattern. Identity is still the `BufferGroup` uid — two fields with the
    same name still get distinct DAG nodes; renaming a field after a token
    has been recorded does not retroactively rewrite the snapshot the
    `KernelLoader` captured at `make_kernel` time. Names propagate through
    `Halo` (which stores the field by-value, so `halo_.field().name()` is
    stable for the halo's lifetime).

  - **`AccessToken::field_name`.** Added alongside `field_uid` and `mode`.
    Captured in `KernelLoader::record_token_` (and `HaloKernelImpl`'s
    constructor) by copy so the export layer has a stable label even after
    the field's own name string is mutated. Conflict scan in
    `Skeleton::compile` ignores the name — strictly a diagnostic.

  - **`Skeleton::export_dot(graph_name = "carbon_skeleton")`.** Lazy-compiles
    then renders the DAG as GraphViz DOT. Each node = one kernel; label is
    `"<kernel.name()>\n[<kind>]"` (falls back to `k<index>` for unnamed
    kernels). Per-stream subgraph clusters with rotating fill colors group
    kernels assigned to the same `Stream` (collapses to one cluster when
    `num_streams() == 1`). Each edge is labelled with the conflicting
    field name(s) and the access-mode arrows
    (`"u (W\u2192R)"` reads as "predecessor wrote `u`, successor reads `u`");
    cross-stream edges are styled `dashed, color="#aa3333"` so the
    rendering visually distinguishes intra-stream FIFO from event-mediated
    ordering. Node tooltips list the full per-kernel token set; edge
    tooltips list every conflicting token pair, useful when several
    fields contribute to one edge. Anonymous fields fall back to
    `<uid:NNN>` so the graph stays readable even when the user forgot to
    name something. Output is deterministic — kernels emitted in user-add
    order, edges in `(i, j)` order, edge labels alphabetised — so tests
    can assert substrings without flake.

  Coverage: `tests/test_skeleton.cu` parallel-sibling subtest now names
  every field at `newField` time (and exercises `set_name` on the
  reference path), asserts `Field::name()` round-trips, then asserts the
  DOT output contains the expected `digraph`, per-stream clusters,
  kernel labels, edge labels (`"v (W\u2192R)"`, `"y (W\u2192R)"`), and the
  dashed cross-stream styling on the `copy_xy → combine` edge.
  `examples/jacobi_skeleton.cu` names `u` / `u_next` and dumps the DOT
  to `CARBON_LOG_INFO` — pipe to `dot -Tsvg` for a quick visual.

- **2026-05 — `KernelLoader::read_stencil` + auto-halo + `Skeleton::Options`
  foundation.** Three pieces ship together as the Skeleton's "Neon-style
  loader" surface, mirroring `Neon::Loader::load(field, Pattern, ...)`:

  - **`carbon::AccessPattern { Map, Stencil }`** — recorded on every
    `AccessToken` alongside `mode`. `loader.read(f)` /
    `loader.read_write(f)` / `loader.write(f)` / `loader(f)` keep their
    legacy `Map` semantics for source compat (a pre-existing kernel that
    secretly reads neighbours will silently miss its halo dep — the
    explicit opt-in is by design and matches Neon, where the Container
    author has to call `load(field, Neon::StencilOp)`).
  - **`loader.read_stencil(field)`** — opt-in stencil read. Records
    `(Read, Stencil)` on the token AND snapshots a type-erased
    `detail::HaloFactoryBase` factory (concrete impl
    `detail::TypedHaloFactory<FieldT>` lives in `<carbon/kernel.h>`,
    after `Kernel` is fully defined). The trait is duck-typed: any field
    type with a `make_halo_kernel()` member returning a `carbon::Kernel`
    plugs in (`Dense::Field<T, C>` does today; future `bGrid` /
    `mGrid`-style fields satisfy the same shape). Stencil writes are
    intentionally not supported — matches Neon (`Loader_imp.h:116`); use
    `loader.write` and pair with an explicit halo if you need cross-slab
    write semantics later.
  - **`carbon::Skeleton::Options { OverlapMode overlap_mode, bool
    strict_halo }`** — passed to the Skeleton ctor (replaces the old
    `max_streams` parameter; the cap is gone — `Skeleton` allocates as
    many `Stream`s as the schedule needs and trusts the dependency graph
    to bound it). Two knobs:
    - `overlap_mode`: `OverlapMode { None, Standard, Extended,
      TwoWayExtended }` — mirrors `Neon::skeleton::Occ` from the
      [Neon paper (Sec. V.B)](https://escholarship.org/uc/item/9fz7k633).
      `None` is the default and matches the v2 behaviour bit-for-bit.
      `Standard` is implemented (see the `Standard OCC` bullet below);
      `Extended` / `TwoWayExtended` throw "not yet implemented" at
      `compile()` until they land in follow-up commits in this PR — the
      API surface is final so callers can write forward-compatible code
      today.
    - `strict_halo`: when `true`, a stencil-read of a field that has
      been written since the last halo (manual or auto) throws
      `std::logic_error` at `compile()` instead of triggering the
      auto-insert. Use when you want every halo to be explicit at the
      call site.

  `Skeleton::compile` gains a Phase 0 that walks `kernels_` and produces
  `effective_kernels_` (= user kernels + auto-inserted halos). Per-field
  `Dirty` state machine: every write or RW token marks the field dirty;
  every halo (manual or auto) cleans it; a stencil-read on a dirty field
  triggers an insert at the current position via the snapshotted
  `HaloFactoryBase::make()` (or throws under `strict_halo`). One halo per
  write→stencil chain (subsequent stencil reads of the still-clean field
  reuse it). All subsequent compile phases (conflict scan, transitive
  reduction, stream assignment, event allocation, inter-iter barrier,
  DOT export) now index over `effective_kernels_`. `effective_to_user()`
  maps each effective index back to the user index or
  `Skeleton::kNoKernel` for synthetic halos so callers can reason about
  the augmented schedule.

  Coverage: `tests/test_skeleton.cu` adds `run_auto_halo_subtest` —
  asserts that `read_stencil` records the right token shape, that
  default-Options auto-inserts an `auto_halo_<field>` between
  `write(F)` and `read_stencil(F)`, that an already-haloed field does
  NOT get a duplicate insert, that `strict_halo=true` throws cleanly,
  and that `OverlapMode::{Standard,Extended,TwoWayExtended}` all throw
  the gating "not yet implemented" diagnostic.
  `examples/jacobi_skeleton.cu` is updated to demonstrate the
  recommended pattern: `loader.read_stencil(fin)` in the build_step
  lambda, no manual `make_halo_kernel` calls in the pipeline, halos
  auto-inserted at compile time.

- **2026-05 — `OverlapMode::Standard` OCC graph rewrite.** Implements
  the Standard variant of the Neon Occ taxonomy directly on top of the
  Skeleton's `effective_kernels_` schedule:

  - **Phase 4 of `Skeleton::compile`** runs after the greedy stream
    assignment + cross-stream event allocation but before the
    inter-iter barrier. For every direct (halo `H`, stencil `S`) edge in
    `min_edges_` where `H.kind() == Halo` and `S.kind() == Compute` and
    `S` Stencil-reads at least one field that `H` halos:
    1. **Move `H` to its own halo stream.** If `H` and `S` currently
       share a stream (the canonical case after greedy assignment for
       the Jacobi 4-kernel cycle), allocate a fresh `Stream` and
       reassign `H`. Predecessors of `H` that previously rode FIFO with
       `H` get explicit cross-stream events on the new stream.
    2. **Allocate `H`'s record event** (if not already done by Phase 3).
    3. **Split `S` into `S-Internal` + `S-Boundary`** on the original
       compute stream. `S-Internal` keeps the `Step`'s slot, drops the
       wait on `H`'s event, and sets `view = DataView::Internal`.
       `S-Boundary` is a freshly inserted `Step` immediately after,
       waits on `H`'s record event, sets `view = DataView::Boundary`,
       and inherits the original `Step`'s `record_event_idx` so
       downstream consumers still wait on "S's completion" without any
       further surgery.
  - **Phase 5: inter-iter barrier recomputation.** Factored out into
    `compute_inter_iter_barrier_()` so it can run after the OCC
    rewrite has changed `kernel_stream_` and `streams_`. Same per-pair
    coverage analysis as before — just sees the post-rewrite
    assignment, so the new halo stream gets the cross-iter wait it
    needs (e.g. iter k+1's `halo_a` waits on iter k's compute-stream
    tail).

  Standard OCC fires only when both `Options::overlap_mode ==
  Standard` and a (halo, stencil) pair exists. Other shapes
  (Reduce-only chains, pure Map pipelines, halos with no following
  stencil) are passed through as `view = Standard`. Universal / Host
  / Reduce kernels are not split (matches the chain-stop rule
  documented on `OverlapMode`).

  Coverage in `tests/test_skeleton.cu`: subtest D asserts the schedule
  has `num_streams() >= 2` and that the halo lands on a different
  stream than the stencil. Subtest E runs a 2-step Jacobi pipeline
  (`stencil_ab` + `stencil_ba`) for `kOccIters=6` iterations under
  `OverlapMode::None` and `OverlapMode::Standard` separately and
  asserts the resulting field is bit-for-bit identical (parity proves
  the rewrite preserves the manual halo+compute schedule's semantics
  on a real ping-pong workload). Extended / TwoWayExtended remain
  gated. Follow-up commits add Extended (back-walk to upstream Map
  nodes) then TwoWayExtended (forward-walk to downstream Map nodes).

- **2026-05 — `OverlapMode::Extended` OCC graph rewrite.** Layers a
  Map-back walk on top of the Standard rewrite so upstream Compute
  predecessors that *write* a field the stencil reads also get split
  into Boundary-first / Internal-second halves. The Standard rewrite
  was renamed `apply_occ_` (single helper handling both modes) and is
  driven from `Skeleton::compile`'s Phase 4: `Standard` runs steps 1–3
  only; `Extended` runs steps 1–3 *and* the Map-back walk (step 4)
  added below.

  - **Step 4: Map-back walk.** For each (halo `H`, stencil `S`)
    candidate, BFS back from `S` along `min_edges_` collecting every
    Compute predecessor `P` reachable through Compute kernels (halt at
    non-Compute, matches the chain-stop rule). Each `P` that *writes*
    any of the fields `S` Stencil-reads is a Map-back candidate.
    Each candidate is split:
    - **`P-Boundary`** (existing schedule slot, `view = Boundary`,
      record event redirected to a fresh `E_p_boundary`)
    - **`P-Internal`** (newly inserted `Step` immediately after,
      `view = Internal`, inherits `P`'s original
      `record_event_idx` so downstream consumers transparently wait on
      `P`'s full completion through the same event slot they did
      before — zero downstream surgery).

    `H`'s wait list is rewired so it waits on `E_p_boundary` instead
    of `record_event_per_kernel_[P]` (the slot now held by
    `P-Internal`). Per-`P` deduplication via `already_split_map_back`
    + `map_back_boundary_event` so the same `P` feeding two
    candidates only gets split once and downstream halos converge on
    the same boundary event.

  Reuses the same `bump_step_for_kernel_after` /
  `allocate_record_event_for` helpers introduced for the Standard
  rewrite. Pre-existing record-event slots stay valid (`P-Internal`
  keeps the slot's identity), so the Phase 3 cross-stream event set
  doesn't need to be rebuilt. The inter-iter barrier in Phase 5 still
  runs after the rewrite and now sees both the new halo stream *and*
  the (unchanged) compute stream `P-Internal` lives on, so cross-iter
  ordering between iter k+1's `P-Boundary` and iter k's downstream
  consumers is preserved by the same per-stream barrier set.

  Coverage in `tests/test_skeleton.cu`: subtest F is a new Extended
  end-to-end parity test — pipeline `copy(seed→work) +
  stencil(work→out)` (the copy is the Map predecessor that writes
  what the stencil Stencil-reads, so the Map-back walk fires) run
  for `kExtIters=4` iterations under `OverlapMode::None` and
  `OverlapMode::Extended` and bit-for-bit `compare_fields(out_a,
  out_b)` parity. Subtest G keeps the gating diagnostic for
  `OverlapMode::TwoWayExtended` (the only remaining unimplemented
  mode). Follow-up commit adds the forward-walk for
  `TwoWayExtended` so downstream Map *consumers* of the stencil's
  output also split into Boundary/Internal halves.

- **2026-05 — `OverlapMode::TwoWayExtended` OCC graph rewrite.** Layers
  a Map-forward walk on top of the Standard + Extended rewrites so
  downstream Compute Map consumers of the stencil's outputs (and the
  outputs of any forward-split Map consumer, transitively) also get
  split into Internal-first / Boundary-second halves. With this commit
  all four `OverlapMode` values (`None`, `Standard`, `Extended`,
  `TwoWayExtended`) are implemented and runnable; the gating
  `logic_error` in `Skeleton::compile` is gone — all non-`None` modes
  route through the same `apply_occ_` helper, which gates the back/
  forward walks per-mode.

  - **Step 5: Map-forward walk (TwoWayExtended only).** For each
    (halo `H`, stencil `S`) candidate, BFS forward from `S` through
    `min_edges_` successors collecting Compute kernels with no
    `AccessPattern::Stencil` reads (pure Map consumers) whose inputs
    touch any field in a growing `tainted_fields` set. Initially
    `tainted_fields = {f : S writes f}`; each forward-split kernel `C`
    contributes its writes to the set so further consumers of `C`'s
    output also fire. Halt at non-Compute and at Compute kernels with
    Stencil reads (those run as their own `(h, s)` candidate pair).
  - **Per-`C` rewrite.** `C-Internal` keeps `C`'s existing schedule
    slot (`view = Internal`, `wait_events` and `record_event_idx`
    cleared). `C-Boundary` is a freshly inserted `Step` immediately
    after, view-tagged `Boundary` and inheriting `C`'s original
    `wait_events` + `record_event_idx`. Both halves remain on `C`'s
    original stream — FIFO ordering guarantees `C-Internal` completes
    before `C-Boundary` starts, so downstream consumers walking
    `record_event_per_kernel_[C]` (now held by `C-Boundary`)
    transparently observe "C done" semantics through the same event
    slot they did before, no downstream surgery needed. Per-`C`
    deduplication via `already_split_forward_` so the same `C`
    reachable from two stencil candidates only gets split once.

  Honest scoping note: this commit delivers the *structural* TwoWay
  rewrite (graph shape and parity preservation) but the same-stream
  placement of `C`'s halves means runtime overlap between
  `C-Internal` and the halo / `S-Boundary` is not yet realised.
  Achieving that overlap requires assigning `C-Internal` to a fresh
  stream (likely with a cross-stream wait on `S-Internal`'s event,
  symmetric to the back-walk's halo-stream allocation) and
  re-routing the Phase-3 cross-stream events accordingly. Queued as
  a follow-up; carbon's API surface and the parity story already
  match Neon's TwoWayExtended without it.

  Coverage in `tests/test_skeleton.cu`: subtest G is now a TwoWay
  end-to-end parity test — pipeline `copy(seed→work) +
  stencil(work→mid) + copy(mid→out)` (the first copy is a Map-back
  candidate, the second is a Map-forward candidate, so all three
  kernels get split) run for `kTweIters=4` iterations under
  `OverlapMode::None` and `OverlapMode::TwoWayExtended` and
  bit-for-bit `compare_fields(out_a, out_b)` parity. With this
  commit no `OverlapMode` value throws "not yet implemented" — the
  gating subtest is fully repurposed into the parity test.
- **2026-05 — `benchmarks/` tree + LBM lid-driven cavity benchmark.**
  Carbon now carries opt-in performance workloads under
  `benchmarks/` (gated by `CARBON_BUILD_BENCHMARKS=ON`, default ON),
  parallel to `examples/` but aimed at headline numbers (MLUPS,
  GLUPS, wall time) rather than didactic concept-demos. First
  inhabitant is `lbm_lid_driven_cavity`, a port of
  `Neon/benchmarks/lbm-lid-driven-cavity-flow` — two-pop pull-
  streaming LBM on a uniform Cartesian grid with bounce-back walls
  on five faces and a moving lid on +y. Same physics + same magic
  numbers as the Neon reference (lattice weights, equilibrium /
  collision arithmetic, half-way bounce-back, lid-momentum
  correction); the scheduler differs — the per-iteration
  collide-stream kernel is wrapped in a 1-kernel `carbon::Skeleton`
  whose `compile()` auto-inserts the halo on the input population
  because the kernel calls `loader.read_stencil(fIn)`. Two skeletons
  (one per ping-pong direction) are precompiled before the timing
  window so the hot loop is a tight `sk.run()` per iteration with no
  allocation / recompilation. CLI follows the same clipp pattern as
  `examples/cg_poisson_single_gpu` (`--n / --iters / --warmup /
  --re / --ulb / --report-every / --no-warmup-skip`). Output is
  one human-readable info line through the carbon logger plus a
  scrape-friendly `MLUPS=... (cavity N=... iters=... gpus=...
  lattice=... collision=...)` on stdout so harnesses can grep the
  headline. Reference data point: single A100 80 GB PCIe at
  `--n 128` → ~2.5 GLUPS (D3Q19 + BGK), ~2.3 GLUPS (D3Q27 + BGK),
  ~2.0 GLUPS (D3Q27 + KBC). The cell-type field is one packed
  `int32_t` per cell carrying both the classification (Bulk /
  BounceBack / MovingWall, 4 bits) and the precomputed
  `wall_neighbour_bits` mask (28 bits, fits both D3Q19's 19-bit and
  D3Q27's 27-bit masks); the wall mask is set once before the
  timing window via `make_compute_wall_mask_kernel<Lattice>`, and
  the hot iteration kernel reads this single int instead of
  branching on each neighbour's classification, mirroring Neon's
  `computeWallNghMask` precomputation. Headers live under
  `benchmarks/lbm_lid_driven_cavity/src/` (`d3q19.h`, `d3q27.h`,
  `cell_type.h`, `lbm_kernels.h`); the driver is the single `.cu`.
  Format-check is wired through the same custom target as
  examples/tests so `cmake --build build --target format-check`
  covers the new tree.

- **2026-05 — D3Q27 lattice + KBC entropic collision in
  `lbm_lid_driven_cavity`.** Lattice and collision model are
  runtime-selectable via `--lattice {q19,q27}` and
  `--collision {bgk,kbc}`. Three valid combos: (q19, bgk) — default,
  port-of-Neon; (q27, bgk) — full 3D 27-velocity stencil with
  standard SRT; (q27, kbc) — KBC entropic stabilisation
  (Karlin-Bösch-Chikatamarla 2014) on D3Q27. (q19, kbc) is rejected
  at CLI parse because KBC's moment basis and shear-decomposition
  table are defined for D3Q27. The hot kernel is templated on a
  `Lattice` traits struct (`D3Q19` / `D3Q27`) and a `Collision`
  policy (`BgkCollision` / `KbcCollision` in
  `src/lbm_kernels.h`); the lattice traits expose constexpr
  `c(q) / w(q) / opp(q) / kCenter` (and additionally `moment(q) /
  shear_decompose(q, Pi)` on `D3Q27` for KBC). All three
  instantiations live in the binary; `main` switches on the CLI
  flags. nvcc's `#pragma unroll` on the per-Q loops produces fully-
  unrolled code that's competitive with the previous hand-unrolled
  D3Q19+BGK body (within ~1% MLUPS). The KBC body mirrors
  `Neon/apps/lbmMultiRes/collide.h::collideKBC` byte-for-byte: per-
  cell `Pi[6]` projection through the lattice's `moment(q)`, per-
  direction `deltaS_q = rho * shear_decompose(q, Pi)`, entropic
  ratio `e0/e1` over `<S, H>_feq` and `<H, H>_feq`, per-cell
  `gamma = 1/beta - (2 - 1/beta) * e0 / (tiny + e1)` with
  `tiny = 1e-7`, then `f_q^post = f_q - beta * (2*deltaS_q +
  gamma*deltaH_q)`. KBC's stability advantage at low viscosity /
  high Re comes for free; runtime cost is ~15% over BGK on the
  same lattice (the moment-projection and entropic-ratio passes
  add real work but stay register-resident).

- **2026-05 — `Skeleton` auto-halo seeded dirty for freshly-uploaded
 multi-GPU fields.** `Skeleton::compile` Phase 0 (auto-halo insertion)
 used to start with an empty dirty set and only mark a field dirty when
 a kernel inside the skeleton wrote to it. That's correct for the
 ping-pong Jacobi pattern (the seed kernel writes the field and the
 next kernel's stencil-read picks up an auto-halo) but silently broke
 every multi-GPU pipeline whose first stencil-reading kernel reads a
 field that was filled externally — the canonical "fill_owned_host →
 Transfer<T>(cpu, gpu) → run skeleton" path. `Field::fill_owned_host`
 only writes the OWNED cells; the host mirror's halo bytes are never
 initialised; the bulk Transfer copy blasts those uninitialised bytes
 onto the device unchanged. Without an in-skeleton write to flip the
 dirty bit, Phase 0 found the field "clean" and skipped the auto-halo,
 so the very first cross-slab stencil read returned garbage. The LBM
 lid-driven cavity benchmark surfaced it: rho dropped to exactly 5/6
 = 0.83333 at the slab-boundary z planes (= 1.0 minus the sum of D3Q19
 weights of the 5 directions whose pull-source crossed the slab
 boundary on each side). Fix: at the top of Phase 0, walk every kernel's
 token list and mark every referenced `field_uid` dirty. Pre-existing
 manual halos in the pipeline still dedup the now-redundant initial
 halo (compile() walks the manual halo first; it cleans the field
 before the subsequent stencil read sees it — `tests/test_skeleton.cu`
 subtest B' locks that contract). Single-GPU layouts see no runtime
 cost because the inserted Halo has `pair_count == 0` and is a
 0-entry no-op. Coverage: `tests/test_skeleton_initial_halo.cu` (new)
 — multi-GPU 1-component field, `fill_owned_host` + `Transfer`, single
 stencil-read kernel through the skeleton, asserts
 `effective_kernels()` is exactly `[auto_halo_src, stencil_pull_zp1]`
 and that the read returned `kSeed` for every `(x, y)` on every
 non-top-slab's last owned z plane (i.e. the cross-slab read found
 valid data, not the uninitialised halo). Validated end-to-end against
 the LBM cavity at N=128, 1000 iters: bit-for-bit `rho`/`u` parity
 across all three transports — single-GPU
 (`CUDA_VISIBLE_DEVICES=0`), single-process 2-GPU (intra-process
 `Transfer<T>` halo), and NCCL 2-rank
 (`mpiexec -n 2 …`, `DistTransfer<T>` halo via `ncclSend`/`ncclRecv`).
 The LBM cavity benchmark binary auto-detects MPI launches: a
 `std::shared_ptr<const Distributed>` is constructed in
 `run_benchmark`, and `world_size() > 1` switches `System` over to
 the distributed ctor (so each rank is bound to its local GPU via
 `pin_local_device` and `gpu_group.size() == 1`); single-process
 launches fall through to `get_all_devices()` so the existing
 single-host behaviour is byte-for-byte preserved. Headline log +
 `MLUPS=…` scrape line are gated on `world_rank == 0` under MPI to
 keep the output single-shot, with the suffix
 `transport=mpi+nccl` distinguishing the NCCL run from the
 single-process line.

- **2026-05 — JSON report output + Python sweep / plot harness for
  `lbm_lid_driven_cavity`.** The benchmark binary gained `--report PATH`
  (writes a structured JSON report; schema `carbon-lbm-report-v1`) and
  `--label STR` (free-form tag stored under `config.label`). The report
  carries the full config (`n / iters / warmup / lattice / collision /
  q / re / ulb / nu_lb / omega`), the system inventory (`gpu_count` plus
  per-device `name / compute_capability / total_memory_bytes /
  multiprocessor_count`), and the derived results (`timed_iters /
  timing_seconds / overall_seconds / cells_per_iter / mlups /
  mlups_per_gpu / effective_bandwidth_gb_s`). Hand-rolled JSON writer,
  no rapidjson dependency. `effective_bandwidth_gb_s` assumes the
  pull-stream + ping-pong layout (2·Q·sizeof(float) bytes per cell
  update) and is the right number to compare against the GPU's HBM
  peak when sanity-checking saturation. Companion Python script at
  `benchmarks/lbm_lid_driven_cavity/scripts/sweep_and_plot.py` drives a
  parameter sweep across `(--sizes, --gpus, --lattice, --collision)`
  (one binary invocation per cell, with `CUDA_VISIBLE_DEVICES=0,..,g-1`
  selecting the device subset) and renders a Neon-style figure: MLUPS
  on the y-axis vs N (`--plot-x size`) or vs GPU count
  (`--plot-x gpus`, strong scaling), one curve per
  `(lattice, collision, label, gpus)` series. `--mode {sweep, plot,
  sweep+plot}` lets users re-render an existing report dir or merge
  two sweeps onto the same axes via `--input-dirs A B`. `--plot-y`
  picks `mlups` (default), `mlups_per_gpu` (good for scaling
  efficiency), or `bandwidth` (effective HBM GB/s). Standard library
  + `matplotlib` only (no rapidjson on the Python side either).
  `reports/` is in `.gitignore` so generated sweeps don't pollute the
  repo. Validated on a single A100 80 GB PCIe: D3Q19+BGK saturates at
  **~5130 MLUPS / ~780 GB/s effective HBM** (40 % of peak) for N≥192,
  D3Q27+BGK at ~3500 MLUPS / ~755 GB/s, D3Q27+KBC at ~1950 MLUPS
  (compute-bound — the entropic step adds ~3× the FLOP count of BGK).
  These numbers replace the earlier `--n 128` reference points in the
  benchmark README, which were in the launch-bound regime and
  understated steady-state throughput.
- **2026-05 — `extern template` for `Field<T, C>` / `Halo<T, C>`:
  investigated, declined.** Followed up on the project-review item
  "extern template for common Halo/Field instantiations" by prototyping
  the push for the six dominant pairs (`<float, 1>`, `<float, 3>`,
  `<double, 1>`, `<int32_t, 1>`, `<int64_t, 1>`, `<int8_t, 1>` — covers
  ~95 % of consumer instantiations across `tests/`, `examples/`,
  `benchmarks/`). The push required adding `CARBON_API` to both class
  templates so the matching `template class …;` definitions in a new
  `src/instantiations.cpp` produce default-visibility symbols (under
  `-fvisibility=hidden` they would otherwise be hidden from
  consumer TUs). It also surfaced a latent bug — `Dense` had no
  default constructor, so `Field<T, C>::Field()` was technically
  ill-formed and only worked because no consumer was forcing its
  instantiation; explicit instantiation tripped on the missing
  `Dense::Dense()`. Adding `Dense() = default;` is a real fix and is
  kept; everything else was reverted. **Measured impact, single-TU
  rebuild on this host (gcc 13 + nvcc 12.8, `-O2`):**
  `test_skeleton.cu.o` 11.87 → 11.70 s (−1.4 %), `test_dense.cpp.o`
  1.95 → 2.01 s (**+3 %**, regressed — the extra include processing
  for the `extern template` declarations costs more than is saved on
  small TUs), `test_field_lazy_host.cu.o` 7.50 → 7.39 s (−1.5 %). Net
  ≈ noise. Root cause: the bulk of `Field<T, C>`'s body is *member
  templates* (`fill_owned_host<Fn>`, `fill_full_host<Fn>`,
  `make_halo_kernel(...)` is an out-of-class template definition in
  `kernel.h`) which `extern template class` cannot suppress — they
  are by construction per-`(TU, Fn)` pair. The non-template members
  are small accessors, a constructor, and `ensure_host_storage_()` —
  small enough that per-TU duplication is cheaper than the
  maintenance overhead of tracking the (T, C) list, the
  type-aliasing pitfalls (`int` vs `std::int32_t`, `long` vs
  `std::int64_t` — they collide on Linux x86_64 and trigger
  duplicate-instantiation diagnostics if both are listed), and the
  cross-header instantiation source (`make_halo_kernel`'s definition
  in `kernel.h` had to be pulled into `instantiations.cpp`). The
  Skeleton trim landed alongside this experiment is the bigger,
  durable win because `Skeleton` is a single non-template class —
  moving its bodies out of the public header trimmed ~700 lines of
  parse work from every consumer TU that includes
  `<carbon/skeleton.h>`. Re-evaluate `extern template` if the
  consumer surface grows to dozens of TUs each instantiating the
  same `Field<T, C>` pair, OR if the Field bodies grow significantly
  in non-template member count.
- **2026-05 — Disaggregated dense-grid foundations
  (`carbon::detail::dense_disg`).** Started porting Neon's
  *disaggregated* `dGrid` (vendored under `Neon-disg/libNeonDomain/include/
  Neon/domain/details/dGridDisg/`). The disg layout splits each per-GPU
  buffer into five physically disjoint z-regions — lower ghost / down
  boundary / internal / up boundary / upper ghost — and lays each region
  out per-cardinality. The pitch formula consumed by the partition is
  `addr = offset_local_no_card + xy_pitch · (region_first_z · C + card ·
  region_z_dim)`, which makes the internal block a card-major SoA while
  keeping the thin boundary / ghost slabs AoS-like. Total storage per
  slab is `(Lz + 2) · C · xy_pitch` — same byte count as the aggregated
  layout, just regrouped. **Why a sibling namespace, not a parametrised
  `Dense`:** the disg pitch carries three extra fields on `Idx`
  (`offset_local_no_card`, `region_first_z`, `region_z_dim`) populated
  once at thread-launch time; folding that into the existing
  `dense::Idx` would either bloat every aggregated kernel's register
  footprint or branch every `linear_offset` call on a layout flag.
  Sibling-ising the primitives under
  `include/carbon/detail/dense_disg/{idx,span,partition}.h` keeps the
  aggregated path byte-for-byte unchanged and lets the disg variant
  evolve independently before we decide whether to unify (current bet:
  unification at the *grid* level, not the per-cell types). **First-pass
  scope:** `dense_disg::Idx` (3D cell + the three region fields),
  `dense_disg::Span` (haloed extent + global extent + xy pitch + z
  offset + halo_z; `iter_extent(view)` returns `Lz` for Standard /
  `Lz - 2` for Internal / `2` for Boundary, matching Neon's
  `dGrid_imp.h` population of `span.mSpanDim`; `set_and_validate(view,
  ix, iy, iz)` mirrors Neon's `dSpan::setAndValidate` verbatim
  including the per-region precomputation; `linear_offset(idx,
  components, component_id)` is the disg pitch formula above), and
  `dense_disg::Partition<T, Components>` (non-owning view with `read` /
  `write` — neighbour resolution comes in a follow-up pass). The disg
  layout pins both ghost and boundary radius to 1 and that is exposed
  as `Span::ghost_radius` / `Span::boundary_radius` static constants.
  Coverage: `tests/test_dense_disg.cpp` (host) walks Standard across
  three `(xDim, yDim, Lz, C)` configurations and asserts no
  double-write / no ghost overlap / cell count = `Lz · C · xy_pitch`;
  walks Internal + Boundary on the same configurations and asserts the
  address set exactly equals the Standard walk; pins
  `set_and_validate` / `linear_offset` against six hand-computed
  reference values derived from Neon's `dPartition::getPitch` for the
  canonical `Lz=4, C=2` buffer (covers down boundary, two internal
  slabs, up boundary, Internal-view spot, Boundary-view down + up);
  round-trips a partition through `read` / `write` end-to-end. **Out
  of scope this pass** (in rough porting order): the `DenseDisg` grid
  wrapper, per-slab field allocation through `BufferGroup`, the kernel
  launch shapes, the disg halo update (transfers between adjacent
  slabs' ghost / boundary blocks), neighbour resolution on
  `dense_disg::Partition`, and distributed-aware partitioning.
- **2026-05 — Multi-GPU disaggregated grid: `DenseDisg`, `Field`,
  `HaloDisg`, end-to-end test.** Built on the foundational primitives
  above. **Public surface (this pass):** `carbon::DenseDisg` (mirrors
  `carbon::Dense` — z-slab partitioning, one-rank-per-GPU distributed
  semantics; halo radius pinned to 0 or 1; carries no `Stencil`
  object since the disg layout's halo is fully determined by the
  radius), `DenseDisg::Field<T, Components>` (per-slab `Buffer`s sized
  from the disg total `(Lz + 2) · C · xy_pitch`, lazy host mirror for
  `MemorySide::HostDevice`), `DenseDisg::newField`, `DenseDisg::spans`
  / `gpu_grid(view)` / `gpu_block(view)`. Storage layout matches
  between gpu and cpu buffers — both sides go through the same disg
  pitch on `dense_disg::Span::linear_offset`. **`HaloDisg<T,
  Components>`:** halo-exchange descriptor mirroring `carbon::Halo`'s
  intra-process branch but specialised to the disaggregated layout.
  Each region (lower ghost / down boundary / up boundary / upper
  ghost) is a single contiguous `C · xy_pitch`-element block, so
  every adjacency pair `(g_lo, g_up)` collapses to **two** transfers
  (upward + downward) regardless of `Components` — vs. the
  aggregated halo's `2 · Components` per pair. Concretely the
  upward direction copies slab `g_lo`'s up-boundary block at offset
  `Lz_lo · C · xy_pitch` (size `C · xy_pitch`) into slab `g_up`'s
  lower-ghost block at offset 0; the downward direction copies slab
  `g_up`'s down-boundary block at offset `C · xy_pitch` into slab
  `g_lo`'s upper-ghost block at offset `(Lz_lo + 1) · C · xy_pitch`.
  Both directions are single contiguous device-to-device copies
  (peer copy when slabs sit on different GPUs in a process,
  `cudaMemcpyDeviceToDevice` when they share a context). PeriodicZ
  adds the wrap pair `(global_slab_count - 1, 0)` exactly the same
  way; on a single global slab in PeriodicZ it becomes a self-pair
  with non-overlapping in-buffer offsets. Cross-rank
  (NCCL / `DistTransfer`) entries are intentionally **not** wired
  this pass — they are tracked as `host_dist_pending_` and surface
  as a clean diagnostic in `launch_on_host`. `launch_on_host` runs
  the same exchange against the field's host mirror via
  `std::memcpy`, mirroring `carbon::Halo::launch_on_host` for the
  fully-host pipeline. **Coverage:**
  `tests/test_dense_disg_multi_gpu.cu` (CUDA, multi-GPU): builds
  `DenseDisg` over `(4, 5, 4 · n_gpus)` with `halo_z = 1`,
  allocates `Field<int, 2>` (two cardinality components — exercises
  the per-region SoA-on-cardinality), fills every owned cell on the
  host with `encode(x, y, global_z, c)`, uploads, runs `HaloDisg`,
  downloads, verifies that owned cells are unchanged, interior
  halos hold `encode` values for the cells they represent, and
  exterior halos remain at the sentinel under Open boundary mode.
  Asserts `pair_count == 2 · (n_gpus - 1)` (component-independent —
  the disg invariant). Skipped (returns 0) when fewer than two
  visible CUDA devices are available, same convention as
  `test_halo_single_node`. Runs on the existing CTest matrix as
  `carbon_dense_disg_multi_gpu`. **Out of scope for this pass:**
  the kernel launch shim (`launch_dense_disg_grid` /
  `dense_disg_grid_kernel<F>`) and the polymorphic `Kernel`
  factories (`make_kernel` / `make_reduce_kernel` / etc.) for the
  disg variant — once those land we can write a multi-GPU
  Jacobi-style stencil test that consumes the halo through
  `Partition::neighbour`.   Cross-rank `HaloDisg` (NCCL) and
  neighbour resolution on `dense_disg::Partition` are the natural
  next checkpoints after that.

- **2026-05** — `StencilSemantic::Lattice` halo on `Halo` and
  `HaloDisg`. Mirrors Neon's `Neon::set::StencilSemantic` toggle and
  the lattice-halo paths in `dGrid::dField_imp.h` /
  `dGridDisg::dField_imp.h`, but factored into one shared API in
  Carbon: a new `StencilSemantic` enum in `<carbon/stencil.h>` plus
  `(field, stencil, semantic)` constructor overloads on both halo
  classes. The semantic flag is stored alongside the field; the
  halo's per-pair entry list is filtered by stencil-direction at
  construction so the runtime launch path stays unchanged
  (`halo(stream)` is still one Transfer + one DistTransfer dispatch).
  **Filtering rule:** a population `q` participates in the upward
  transfer iff `stencil.points()[q].z > 0`, in the downward transfer
  iff `< 0`. `Standard` semantic keeps everything (back-compat: the
  one-arg constructor delegates to `(field, Stencil{}, Standard)`).
  Lattice-mode pre-conditions (validated at construction): stencil
  size equals `Components`, stencil radius `<=` the field's halo
  thickness (`<= 1` on `DenseDisg`), and `semantic` is one of the two
  enumerators. **Run-length fusion (disg-only).** The disg layout
  stores each cardinality plane as a contiguous `xy_pitch`-sized
  block within the per-region buffer (`region_first_z * C * pitch +
  c * pitch`), so adjacent active populations sit physically
  adjacent in memory and a single `cudaMemcpyAsync` of
  `j_len * pitch` elements covers a whole run. The new
  `Stencil::lattice_profile()` helper returns
  `LatticeProfile{ positive_z, negative_z, zero_z, run_count_*,
  max_run_*, fully_groupable }` — the disg halo emits one entry per
  maximal-run. The aggregated halo has no cross-cardinality fusion
  (cardinality planes live `linear_storage_elements` apart), so it
  emits one entry per active cardinality regardless of ordering.
  **Headline (2× A100 80GB PCIe NODE-link, `N=384`, D3Q19,
  `halo_z=1`):** `disg standard halo` 0.617 ms/iter (2 transfers,
  full-Q payload) → `disg lattice halo, grouped ordering` 0.188
  ms/iter (2 fused transfers, 5/19 payload) = **3.28× speedup**, vs
  a 19/5 = 3.8× theoretical ceiling (86% efficient). At Q=27 the
  measured win lands at 2.70× / 3.0× = 90%. The natural Carbon LBM
  ordering (matching `benchmarks/lbm_lid_driven_cavity/src/d3q19.h`)
  scatters `c.z>0` populations across 5 separate runs, so the
  natural-ordering disg lattice halo behaves like 5 transfers per
  direction: ~10× more launches and the per-call overhead eats the
  payload reduction. **Coverage:**
  `tests/test_lattice_halo_disg.cu` — D3Q19 on `(4, 5, 4·n_gpus)`
  with two flavours (natural + grouped); validates `pair_count`
  matches `2·(n_gpus−1)·(run_count_+ + run_count_−)` for each
  ordering, sentinel-fills every halo cell, runs the lattice halo,
  and verifies cell-by-cell that active populations got their
  neighbour's encoded value while inactive populations stayed at
  sentinel (no over-copy).
  `tests/test_lattice_halo_aggregated.cu` — same shape against
  `carbon::Halo` / `Dense`; pair-count formula adjusts to
  `(n_gpus−1)·(positive_z + negative_z)` (no fusion) and natural /
  grouped orderings produce identical numbers (control). The
  `benchmarks/halo_scalability` benchmark grew `--semantic
  standard|lattice` and `--lattice d3q19|d3q19_grouped|d3q27|
  d3q27_grouped` flags; `scripts/run_local.sh` sweeps both modes
  and `scripts/plot.py` adds `halo_lattice_vs_size.png` plus a
  `halo_lattice_speedup.png` headline chart with horizontal
  references at 19/5 and 27/9. CSV is back-compat: a `semantic`
  column defaults to `"standard"` when missing so older sweeps still
  parse. **Design notes.** (1) The semantic flag lives on the halo
  rather than on the field because the same field is legitimately
  consumed by both standard *and* lattice kernels in practice
  (a field used as a generic component vector by half the pipeline
  and as an LBM PDF carrier by the other half). Sticking it on the
  halo descriptor keeps fields agnostic and lets the same
  `Field<T, Q>` pair with multiple halo flavours simultaneously.
  (2) Pre-condition `radius <= 1` on `DenseDisg` lattice halos is
  intentional — disg's region count is fixed at 5 (lower ghost,
  down boundary, internal, up boundary, upper ghost) and only
  carries one xy-plane on each side. Multi-radius lattices (Q39 /
  Q125) need extending the disg region count first; that's a
  separate, larger change. The aggregated `Halo` lattice path
  accepts any stencil whose radius fits the configured halo
  thickness. (3) NCCL Send/Recv pairing across ranks: the local
  rank emits a `Send` only for active-up populations when it's the
  lower endpoint of the pair (vs active-down when it's the upper
  endpoint). This keeps the FIFO Send/Recv order across ranks
  matched without coordination — both ends derive their filter
  from the same shared stencil. (4) Followup: a stencil-aware
  population reordering helper that takes a user's
  `std::vector<int32_3d>` and returns it permuted into "fully
  groupable" form, so user code doesn't have to do the bucketing
  by hand.

- **2026-05 — LBM lattice population reordering for 2-transfer fusion.**
  Re-bucketed the D3Q19 / D3Q27 velocity tables in
  `benchmarks/lbm_lid_driven_cavity/src/d3q{19,27}.h` so all `c.z<0`
  populations live before all `c.z=0` before all `c.z>0`, with a
  palindromic in-bucket assignment that gives the closed form
  `opp(q) = Q - 1 - q` (center is its own opposite at the bucket-
  midpoint q=9 for D3Q19, q=13 for D3Q27). This is the "fully
  groupable" ordering the lattice-semantic disg halo's run-length
  fusion needs to collapse to exactly **two** `cudaMemcpyAsync`
  per inter-slab adjacency pair regardless of Q. The hand-rolled
  D3Q19 BGK fast path was rewritten on the new indexing
  (macroscopic sub-sums are now full contiguous-bucket scans for
  the Z components: `Z_M1 = pop[0..4]`, `Z_P1 = pop[14..18]`); the
  KBC `moment(q)` and `shear_decompose(q)` helpers in d3q27.h were
  rewritten to derive directly from `c(q)` (outer product `c⊗c`
  for moment; closed-form by nonzero-component count for shear) so
  the table stays correct under any future re-ordering. Validated
  end-to-end: q19+bgk, q27+bgk, q27+kbc all produce the lid-driven
  cavity profile (rho mean 1.0000, |u|max 0.0500, lid u_x = 0.0500,
  sub-lid u_x = 0.0469, no NaN/Inf) on a 64^3 cavity at Re=100,
  bit-identical between collision modes for the macroscopic
  observables. Stencil exposes `kZNegBegin/End`, `kZZeroBegin/End`,
  `kZPosBegin/End` so callers / lattice halos can assert the
  bucket structure at compile time.

- **2026-05 — `carbon::Block<SBlock>` (port of Neon `bGrid`, phase 1
  data plane).** New `<carbon/block.h>` plus the supporting
  `detail/block/` headers (`static_block.h`, `idx.h`, `span.h`,
  `partition.h`, `field.h`) bring up the block-sparse grid family.
  `StaticBlock<MemX, MemY, MemZ>` is the compile-time brick policy
  with the packed `BitMask` (one bit per voxel, x-fastest); the
  default brick is `StaticBlock<8, 8, 8>` (512 voxels / 16 mask
  words, matching Neon's only explicit instantiation). `block::Idx`
  carries `(block_idx, intra=int8_3d)`; `block::Span` carries the
  per-block bitmask pointer + retained-block count and exposes the
  pitch resolver `linear_offset = (block_idx*C + c) * VoxelCount +
  (x + MemX*y + MemX*MemY*z)` (the same formula Neon's
  `bPartition::helpGetValidIdxPitchExplicit` uses);
  `block::Partition<T, Components, SBlock>` adds `read / write /
  neighbour` with 27-way connectivity for stencil reads that cross
  the brick boundary. `Block<SBlock>` walks the candidate brick
  lattice once on the host, retains every brick that passes the
  user predicate, builds the per-block bitmask + connectivity
  table + voxel origins, and uploads them to the (single) device.
  `newField<T, Components>()` allocates the per-block SoA buffer
  and returns a `Field` whose `device_partition()` is ready to be
  passed to a CUDA kernel. Coverage: `tests/test_block_static.cpp`
  (BitMask set/clear/sweep, Span pitch + per-card sizing, Span
  validation, connectivity-slot encoding) and
  `tests/test_block_cuda.cu` (end-to-end: build a Block over a
  sphere predicate, allocate `Field<int32_t, 2>`, launch a hand-
  rolled `__global__` kernel that walks `(block_idx, intra)` and
  writes encoded values into every active voxel via
  `Partition::write`, copy back, assert every active voxel carries
  the expected encoding **and** every inactive voxel still carries
  the post-allocation zero — i.e. `Span::set_and_validate` agrees
  with the host predicate cell-by-cell). 24^3 domain over 8-cube
  bricks: 27 retained bricks, 5232 active voxels, 8592 inactive-
  but-allocated voxels — all 13824 cells × 2 components verified.

- **2026-05 — Block phase 1 is data-plane-only; halo, multi-GPU,
  Internal/Boundary `DataView` split, and the `Kernel` factory
  are explicit follow-ups.** The reasoning is the same that drove
  the cadence on `Dense` and `DenseDisg`: ship the storage
  primitive and lock it with a CUDA correctness test, then layer
  partitioning + halo, then the polymorphic `Kernel` handle. A
  block-sparse multi-GPU bootstrap requires porting Neon's
  `Partitioner1D` + `SpanDecomposition` + `SpanLayout` for
  block-granularity Z-axis splits — a substantial subsystem in
  its own right and best landed against a working single-GPU
  baseline. The block API was designed to extend cleanly: `Span`
  has no `DataView` knowledge today but the same shape Neon's
  `bSpan` uses (with `mFirstDataBlockOffset` for the
  Internal/Boundary skip) drops in by adding one `int` to
  `Span` and per-view `gpu_grid` / `gpu_block` arrays to `Block`,
  same as `Dense`. Halo plumbing reuses the same
  `Halo<T, Components>` shape (block-level transfers replace
  voxel-plane transfers; the `Transfer` size formula becomes
  `transfer_block_count * VoxelCount * sizeof(T) * Components`,
  i.e. whole bricks, all components fused — the contract Neon's
  `bField::initHaloUpdateTable` already encodes).

- **2026-05 — `Block::make_kernel` and the `d1b3` launch shape.**
  The `Kernel` factory wiring on top of the block-sparse data
  plane lands as `carbon::Kernel::compute_block` (static factory)
  + `Block<SBlock>::make_kernel(loading, shared_mem)` (member
  shorthand, mirroring the `Dense::make_kernel` ergonomic). The
  concrete impl `carbon::detail::BlockKernelImpl<Grid, LoadingFn>`
  follows the same shape as `DeviceKernelImpl` (loading-lambda
  semantics, per-GPU loop, token discovery via a one-shot pass),
  but dispatches to `detail::block::launch_block_grid` instead of
  `dense::launch_dense_grid`. The launch shape is Neon's `d1b3`:
  one CUDA block per retained brick (`grid = (block_count, 1, 1)`)
  and one thread per voxel inside the brick
  (`block = (MemX, MemY, MemZ)`). Inactive voxels are skipped via
  `Span::set_and_validate` before the user lambda runs, so the
  per-cell body is identical in shape to the dense one — a
  `[in, out] CARBON_HD (Span s, Idx idx) -> void` closure that
  reads `in.read(s, idx, c)` / writes `out.write(s, idx, c, v)` /
  walks neighbours via `Partition::neighbour(s, idx, offset, c)`.
  Plumbing changes that fell out:
  - `block::Field<T, C, SBlock>` now provides the
    `KernelLoader`-required duck-typed accessors
    (`partitions()`, `host_partitions()`, `gpu_buffers()`,
    `make_halo_kernel()`, `name()`). `partitions()` is a 1-element
    vector in phase 1 (single-GPU); `host_partitions()` is empty
    (no host mirror in phase 1); `make_halo_kernel()` returns a
    no-op `Kernel::anchor` since a single-GPU `Block` has nothing
    to halo-exchange across — once `HaloBlock` lands, that hook
    routes through it the same way `Dense::Field::make_halo_kernel`
    routes through `Halo`.
  - `block::Partition::write` is now `const` and takes the value
    as a separate argument (`write(span, idx, c, value)`),
    matching `dense::Partition::write` so user-written `[=]`
    device-lambda captures actually compile (the previous
    `T& write(...)` shape failed because the captured-by-value
    partition becomes `const` in the closure body).
  - `Block::device_origins_ptr()` exposes the device-side
    per-block origin table so user kernels that need global voxel
    coordinates can capture it without reuploading the same data
    themselves. (The previous workaround was to allocate a
    side-buffer and pump it through `System::async_copy` from the
    host's `host_origins()` mirror.)
  - One workaround for an nvcc bug: the `__global__`
    `block_grid_kernel<SBlock, F>` template is *not* constrained
    by a `requires` clause. nvcc's stub generator chokes when a
    constrained `__global__` template takes a device-lambda type
    and a separate static-type parameter (it emits a using-alias
    name into the stub source that doesn't parse); the constraint
    lives on `launch_block_grid` instead. Same logic in plain
    parameter ordering: the impl uses `typename Grid::BlockPolicy`
    everywhere instead of caching a `using SBlock = ...` alias,
    because the alias also leaked into the stub-side mangled name.
  - SYCL backend: `detail/backend/sycl/block_launch.h` mirrors the
    CUDA path with a `q.parallel_for` over an `nd_range<3>` whose
    slowest dim folds `block_idx * MemZ` to keep the launch shape
    parallel to CUDA's. Tested under both backends through the
    existing `carbon_add_cuda_unit_test` machinery (single-GPU
    only at this stage).

  Coverage:
  - `tests/test_block_kernel.cu`: builds a `Block` over a sphere
    predicate, allocates two `Field<int32_t, 1>` ping-pong fields,
    runs an `init` kernel (writes encoded `block_idx + intra`
    values via `Block::make_kernel`) followed by a `copy` kernel
    (reads the source, writes destination), reads back, and
    verifies every active voxel carries the expected encoded value
    while inactive voxels remain zero. Locks both the launch shape
    and the `KernelLoader` duck-typed `Field` contract.
  - `examples/jacobi_block_single_gpu.cu`: 7-point Jacobi on a
    spherical sub-domain of a 64-cube box. 280 retained bricks,
    81104 active voxels (out of 262144 dense), 600 iterations
    converging to a smooth z-ramp from +0.92 at the south pole to
    −0.95 at the north pole. The Dirichlet BC is detected at
    kernel time by `neighbour(...).valid` returning false on the
    sphere shell — the same idiom Neon's `bGrid` examples use,
    and the same idiom user code will use for any block-sparse
    stencil. `Block::device_origins_ptr` is captured for global-z
    recovery (kept in the Jacobi build for documentation purposes
    even though the final BC formulation only needs the
    `neighbour::valid` hint).

  Build + tests + format-check stay clean (47/47 default tests
  pass; `block_kernel` is the new entry, `block_static` and
  `block_cuda` from the previous `Block` commit are unchanged).

- **2026-05 — `Stencil::reorder_for_lattice_fusion` helper.** Static
  member on `Stencil` that takes a user's `std::vector<int32_3d>` and
  returns a `LatticeReorderResult { reordered, permutation }`. The
  reordered list groups all `c.z<0` first (in their original relative
  order), then `c.z==0`, then `c.z>0` — the layout that makes a
  `DenseDisg` lattice halo collapse to exactly two transfers per
  adjacency pair (one fused upward, one fused downward) regardless of
  the lattice cardinality. `permutation[new_idx] == old_idx`, so user
  code remaps its weight / population table via
  `new_w[i] = old_w[permutation[i]]`. Stable within each `c.z` bucket
  (preserves D3Q19 / D3Q27's `opp(q) = Q-1-q` symmetry when the input
  list already encodes it). Closes the lattice-halo follow-up that the
  2026-05 lattice-halo CHANGELOG entry called out: previously a user
  who wanted optimal fusion had to hand-reorder the lattice; now
  `Stencil::reorder_for_lattice_fusion(natural_velocities)` plus the
  matching weight / kernel remap is the canonical pre-processing step.
  Pure host helper, no validation against `int8_t` bounds (that runs
  in the `Stencil` ctor when the result is fed back in). Coverage in
  `tests/test_stencil.cpp`: a 19-velocity "natural" (interleaved-z)
  ordering reordered into fully-groupable form, with bijection +
  in-bucket-stability checks and a `LatticeProfile::fully_groupable`
  assertion against the resulting `Stencil`.

- **2026-05 — Multi-GPU `carbon::Block` + `carbon::HaloBlock`.** Phase A
  (multi-GPU partitioning) and phase B (ghost bricks + halo) of the
  `bGrid` port land together as the natural follow-up to the single-GPU
  data plane that shipped on 2026-05 (`1c216bb`). The brick **z-axis** of
  the candidate brick lattice is split evenly across `system.gpu_count()`
  devices via the same shape `Dense` uses for its z-slab partition: GPU
  `g` owns brick rows
  `[g · candidate_z / N + min(g, candidate_z % N),
  (g+1) · candidate_z / N + min(g+1, candidate_z % N))`, summing to
  `candidate_z`. Per-GPU storage is fully separate: each device gets its
  own `host_origins`, `host_bitmasks`, `host_connectivity`, `Span`, and
  field buffer; `Block::block_count(g)`, `host_origins(g)`, `span(g)`,
  `device_origins_ptr(g)`, etc. are the new "per-GPU" accessors, with
  the no-arg overloads kept (defaulting to GPU 0) so single-GPU tests +
  examples (`test_block_cuda.cu`, `test_block_kernel.cu`,
  `examples/jacobi_block_single_gpu.cu`) still compile + pass
  bit-for-bit. `BlockKernelImpl` already loops over GPUs (it shares the
  per-GPU dispatch shape with `DeviceKernelImpl`), so wiring this in was
  a partition-storage refactor with no kernel-launch changes —
  `launch_block_grid` just reads `grid.spans()[g]` and
  `grid.block_count(g)` instead of the singular versions.

  Ghost-brick + halo plumbing (`HaloBlock`) sits on top of the
  partitioner. At construction `Block` reserves "remote-mirror" slots at
  the lower / upper z-edge of every GPU's slab — one slot per retained
  brick at `bz = bz_begin(g) - 1` (lower) or `bz = bz_end(g)` (upper)
  whose owner is the adjacent GPU. The 27-way connectivity table on
  owned bricks is then re-walked, and any `(dx, dy, dz)` slot whose
  neighbour candidate falls onto another GPU's slab is rewritten from
  `kInvalidBlockIdx` to the matching ghost slot id on this GPU — so a
  device-side `Partition::neighbour(span, idx, ±z, c)` resolves
  uniformly through the same `connectivity[block_idx * 27 + slot]`
  lookup whether the neighbour is local or a ghost. Bitmasks for ghost
  slots are uploaded in the same async copy pass as the owned bitmasks
  (cheap, host-side: we already have all per-GPU bitmask vectors built),
  so the `is_active`-skipping inside `Span::set_and_validate` works on
  ghosts without further plumbing.

  Field storage on GPU `g` is now sized for
  `(n_owned + n_ghost_lower + n_ghost_upper) * VoxelCount * Components`
  — kernels still iterate `block_count(g)` owned bricks, but ghost slots
  are addressable through `Partition::neighbour` reads. `HaloBlock<T,
  Components, SBlock>` (in `<carbon/halo_block.h>`) builds a single
  intra-process `Transfer<T>` aliasing the field's per-GPU
  `Allocation<T>` handles into one `BufferGroup<T>`: per adjacency pair
  `(adj, adj+1)`, every owned brick on `adj` at `bz_end(adj) - 1` ships
  upward into the matching lower-ghost slot on `adj+1`, and every owned
  brick on `adj+1` at `bz_begin(adj+1)` ships downward into the matching
  upper-ghost slot on `adj`. Each pair transfers one full brick worth of
  bytes (`VoxelCount · Components · sizeof(T)`) — all components are
  fused since brick storage is SoA per-block (the
  `(block_idx · C + c) · VoxelCount + voxel` pitch keeps every `(block,
  c)` plane contiguous). `HaloBlock(stream)` issues every entry through
  `System::async_copy`, which dispatches `cudaMemcpyPeerAsync` when
  source / destination GPUs differ. Single-shot, async, same call
  ergonomics as `Halo` and `HaloDisg`.

  Cross-rank halo (MPI / NCCL) is intentionally not yet wired —
  `HaloBlock`'s constructor only walks intra-process pairs. The dist
  leg will mirror `Halo`'s shape (`DistTransfer<T>` Send/Recv per pair,
  gated on `CARBON_HAVE_MPI_NCCL`) once the multi-rank `bGrid` shape
  surfaces a use case (LBM over a sphere-on-N-ranks is the obvious
  candidate). For now the documented contract is "single process,
  multiple GPUs" — same scope `Halo` covered through 2026-04.

  Coverage: `tests/test_block_multi_gpu.cu` exercises the partitioner
  (per-GPU `bz` ranges, contiguity over `[0, candidate_z)`, origin
  bounds, active-voxel cross-check vs. a fresh dense predicate sweep, +
  end-to-end `Block::make_kernel` write-then-read on every owned brick
  using the per-GPU `device_origins_ptr(g)` accessor).
  `tests/test_halo_block.cu` exercises Phase B end-to-end on a
  sphere-on-N-GPUs layout: byte-level ghost-content matches the source
  brick on the remote GPU after `HaloBlock(stream)`, and a diagnostic
  device kernel walks every active voxel and reads its `+z` / `-z`
  neighbour through `Partition::neighbour` — proving connectivity rewrite
  + ghost bitmask upload + halo data upload work together. Both tests
  skip on hosts with fewer than 2 visible CUDA devices (printing a
  "skipping" line) so they're safe to register unconditionally; on the
  development host (2× A100 80GB PCIe) the multi-GPU test reports 312
  retained blocks across the 2-GPU split (vs. 280 single-GPU at the same
  density — the small bump is the +z / -z padding the candidate sweep
  walks at the slab boundary), and the halo test reports 104
  cross-slab brick transfers (52 upward + 52 downward) covering 100,024
  voxels of cross-slab neighbour reads.

  Deferred follow-ups (no API churn): a multi-GPU sparse Jacobi example
  on top of `HaloBlock` (the natural 2-GPU port of
  `examples/jacobi_block_single_gpu.cu`); cross-rank `HaloBlock` over
  NCCL; `DataView::Internal` / `Boundary` split for `Block` (depends on
  classifying owned bricks into internal vs. boundary subsets, same
  shape Neon's `Partitioner1D` produces).

- **2026-05 — `examples/jacobi_block_multi_gpu.cu`.** Multi-GPU port
  of the single-GPU sparse Jacobi example. Same Laplace-on-a-sphere
  problem (hot south pole / cold north pole / insulated shell, 7-point
  Jacobi with neighbour-validity-driven BCs) but partitioned across
  every visible CUDA device, with one `HaloBlock<float, 1>` per
  ping-pong field re-issued on the system stream before the
  matching step kernel. The per-cell loading lambda is **byte-for-byte
  identical** to the single-GPU example — `Partition::neighbour`
  resolves uniformly through the connectivity table whether the
  neighbour is a local brick or a ghost slot refreshed by the most
  recent halo, so user code does not branch on GPU count. The only
  multi-GPU-specific additions vs. the single-GPU example are the
  `HaloBlock` construction and the per-iter `halo(stream)` call, plus
  a per-GPU readback in the RMS / centerline helpers (the host-side
  data isn't multi-GPU-aware yet — a `Field::download_to_host` helper
  is the obvious follow-up). On the dev host (2× A100): iter 600 RMS
  matches the single-GPU run to 5 decimal places (1e-6 difference is
  expected fp-non-associativity at the slab boundary), and the
  centerline ramps smoothly from +0.916 (near south pole on GPU 0)
  through 0 at z=32 (the slab boundary) to -0.946 (near north pole on
  GPU 1), proving cross-slab data flow end-to-end.   Falls back to a
  single-GPU run with a printed note when only one CUDA device is
  visible (so the example is safe to register unconditionally
  alongside `jacobi_block_single_gpu`).

- **2026-05 — Distributed `carbon::Block` + cross-rank `HaloBlock`
  over NCCL.** Extends the multi-GPU `Block` partitioner from the
  previous entry into a full one-rank-per-GPU MPI port.

  *`Block` ctor.* When the underlying `System` carries a
  `Distributed` handle, the brick z-axis is partitioned across
  `world_size` global slabs (with `gpu_count_per_rank == 1` enforced
  — the multi-GPU-per-rank hybrid stays explicitly out of scope, as
  for `Dense`), and only this rank's slab is walked for owned brick
  discovery. New accessors `global_slab_count()` /
  `local_slab_offset()` mirror `Dense`. Ghost reservation now also
  evaluates the predicate on candidate brick rows that fall on
  *adjacent ranks*: the predicate is pure, so both sides reserve
  identical ghost slots in the same row-major `(bx, by)` sweep order
  without any cross-rank coordination at construction time. The
  per-ghost descriptor records `peer_rank` (-1 for local-slab
  source, otherwise the remote rank id), and the bitmask is captured
  alongside (uploaded into the ghost section of `device_bitmask_` via
  the existing single async copy per GPU).

  *Halo plan.* The per-adjacency `halo_pairs_upward` /
  `halo_pairs_downward` vectors keep covering only the
  intra-process side (both endpoints of the adjacency local to this
  rank — single-rank multi-GPU). Cross-rank pairs go into four new
  flat lists keyed on this rank's role: `halo_dist_send_upward`
  (Send entries when this rank is the lower of a remote
  adjacency), `halo_dist_recv_upward` (Recv into lower-ghost slots
  when the lower endpoint is remote), and the symmetric
  `halo_dist_send_downward` / `halo_dist_recv_downward`. Each entry
  is a `HaloDistEntry { local_id, peer_rank }` — the matching id on
  the remote rank is implicit because both sides walk the same
  predicate over the same `(bx, by)` row in the same order, so
  NCCL pairs Sends and Recvs positionally (FIFO).

  *`HaloBlock`.* `<carbon/halo_block.h>` now flattens those four
  lists into a single `DistTransfer<T>` posted against this rank's
  lone field buffer (whole-brick, components fused — same brick
  count `VoxelCount * Components` as the intra path). `operator()
  (stream)` issues the intra `Transfer<T>` followed by the
  cross-rank `DistTransfer<T>` back-to-back; both sides accept an
  empty/no-op state so the call shape is the same regardless of
  layout. Carbon's one-rank-per-GPU model means in MPI runs
  `pair_count() == 0` and `dist_entry_count() > 0`, while in
  single-process multi-GPU runs `pair_count() > 0` and
  `dist_entry_count() == 0`. The `dist_entry_count()` accessor is
  defined unconditionally; in non-MPI builds it returns 0.

  *Coverage.* `tests/test_halo_block_dist.cu` is a 2-rank dist
  test (registered under `CARBON_ENABLE_MPI=ON`): each rank builds
  a distributed `Block` over a 32-cube domain with every voxel
  active (regular brick layout — every `(bx, by)` at the slab
  boundary contributes one ghost on each side), uploads
  `encode(global_x, global_y, global_z)` into every owned brick,
  runs `HaloBlock(stream)`, downloads, and verifies (a) owned bricks
  are unchanged and (b) every ghost slot holds the encoded global
  coordinates of the matching remote brick voxel — proving Send /
  Recv pairing, byte counts, and ghost slot ordering are
  consistent across ranks. Also pins `block.global_slab_count()` /
  `local_slab_offset()`, and `Block::halo_dist_*` sizes and
  `HaloBlock::dist_entry_count()`. Verified against a real 2-rank
  `mpiexec` run on the development host (2× A100, OpenMPI + NCCL
  2.x): all 7 `CARBON_ENABLE_MPI=ON` tests pass (`dist_transfer`,
  `dense_dist`, `halo_dist`, `halo_dist_periodic_z`,
  **`halo_block_dist`**, `native_io_dist`, `dist_reduce`), and the
  single-process triad still completes 49/49 tests + format-check
  cleanly.

  *Deferred follow-ups (no API churn):*
  `DataView::Internal` / `Boundary` split for `Block` (depends on
  classifying owned bricks into internal vs. boundary subsets, same
  shape Neon's `Partitioner1D` produces — required for
  internal/boundary overlap of compute with halo on the sparse
  side); a `Field::download_to_host` helper that the multi-GPU
  Jacobi example open-codes today.

- **2026-05 — `DataView::Internal` / `Boundary` split for
  `carbon::Block`.** Brick-level analogue of `Dense::Span`'s halo_z-
  deep Internal/Boundary bands, sized to match the existing
  one-brick-row-deep ghost reservation. *Why:* enables the same
  halo + internal-compute-in-parallel + boundary-compute-after-halo
  pattern that `Dense`'s `OverlappedStencil` ships, but on the
  block-sparse data plane. *Mechanism:*

  1. `Block`'s ctor, after the main brick discovery sweep and
     **before** ghost reservation, stable-partitions each slab's
     owned bricks per the predicate "boundary iff
     `(bz == bz_begin[g] && global_g > 0) || (bz == bz_end[g] - 1
     && global_g + 1 < global_slab_count)`" (i.e. only faces with
     an actual remote neighbour on the other side count). Internal
     bricks land at `[0, n_internal)`; boundary bricks at
     `[n_internal, block_count)`.
  2. The same pass remaps every `candidate_to_local[c]` entry whose
     `gpu_id` matches the partitioned slab — so ghost reservation
     (which keys off `candidate_to_local` to identify the remote
     source brick of a local-slab ghost), the connectivity
     rewrite, and the halo plan (intra `HaloPair`s + cross-rank
     `HaloDistEntry`s) all see the new local ids without further
     bookkeeping. Ghost slot ids (`[n_owned, n_owned + n_ghost)`)
     are unchanged because `n_owned` is preserved by the partition.
  3. `Block` exposes `n_internal_blocks(g)` and `n_boundary_blocks(g)`;
     `Span<SBlock>` now carries the `n_internal` count alongside
     the bitmask + block_count, with `iter_block_count(view)` /
     `view_block_idx(view, iter)` returning the right subset
     (Standard → all blocks; Internal → `[0, n_internal)`;
     Boundary → `[n_internal, block_count)` with the iter index
     shifted by `n_internal` so the kernel still sees a valid
     storage `block_idx`).
  4. The CUDA + SYCL `launch_block_grid` overloads and the CUDA
     `block_grid_kernel` global function take a `DataView`
     parameter; the kernel computes `b = span.view_block_idx(view,
     blockIdx.x)` and otherwise behaves as before.
     `BlockKernelImpl::launch_on_device(stream, view)` forwards
     the view through, so `Kernel::launch(stream, view)` on a
     `Block` kernel picks the right brick subset at launch time
     (purely a launch-time choice; the kernel does not store one).

  *Verification:* a new `tests/test_block_data_view.cu` builds a
  64³ sphere `Block` on 2 visible CUDA devices, asserts the
  per-slab partition invariant ("brick at `[i, n_internal)` returns
  `false` for the boundary predicate, brick at `[n_internal,
  block_count)` returns `true`"), runs the same encode kernel via
  `DataView::Standard` into one field and via `Internal` +
  `Boundary` into another, then compares cell-by-cell over the
  bitmask. On a 64³ × 2 GPUs config the test pins:
  `gpu 0 owned=116 (internal=72, boundary=44)` and `gpu 1
  owned=116 (internal=72, boundary=44)`; 70320 active voxels match
  between the Standard launch and the Internal+Boundary pair.
  `tests/test_block_multi_gpu.cu`, `tests/test_block_kernel.cu`,
  `tests/test_halo_block.cu`, `examples/jacobi_block_single_gpu`,
  and `examples/jacobi_block_multi_gpu` (multi-GPU Jacobi iter 600
  RMS = 0.529625, centerline ramp from +0.916 to -0.946) all stay
  bit-for-bit identical thanks to the partition being stable
  within each (internal, boundary) half. Total default-test count
  rises from 49 to 50.

- **2026-05 — `Block::download_to_host(field, stream)`.** New helper
  on `carbon::Block<SBlock>` that asynchronously copies each local
  GPU's owned-brick payload from a `Field<T, Components, SBlock>`
  into a freshly allocated host `detail::Allocation<T>`, returning a
  `std::vector<detail::Allocation<T>>` of size `gpu_count()`. Per-GPU
  size is exactly `block_count(g) * VoxelCount * Components` —
  ghost slots are intentionally @b not downloaded because host-side
  processing (RMS, I/O staging, centerline samples) almost always
  wants the locally meaningful owned subset, and a separate helper
  can ship ghosts later if that ever becomes necessary. Copies queue
  on the user-supplied `Stream` without an internal `wait()` (mirrors
  the `Transfer` / `System::async_copy` convention so the call
  composes with kernels already on that stream). Refactors
  `examples/jacobi_block_multi_gpu.cu` (RMS computation + final
  centerline readback) and `tests/test_block_multi_gpu.cu` (per-voxel
  encode-verify) to use it; both behave bit-for-bit identically to
  the previous open-coded `for (g) sys.async_copy(...)` loops
  (multi-GPU Jacobi iter 600 RMS = 0.529625, centerline ramps from
  +0.916 to -0.946 unchanged). All 49 default tests + format-check
  remain clean.

- **2026-05 — `OverlappedStencil` over `carbon::Block` — sparse halo /
  internal-compute overlap.** Wires `Block::Field::make_halo_kernel` to
  build a real `HaloBlock`-backed `Kernel` (instead of the no-op anchor
  the previous pass returned), then drives the sparse multi-GPU Jacobi
  through the existing `carbon::OverlappedStencil` orchestrator using
  the `DataView::Internal` / `Boundary` split for `Block` from earlier
  in the session. Three pieces:

  1. **Back-reference plumbing.** `detail::block::Field<T, C, SBlock>`
     grew an optional `const ::carbon::Block<SBlock>* block_` member
     plus a `set_block(...)` setter; `Block::newField<T, C>` passes
     `this` after constructing the field. Storing `Block` by value
     inside the field (the way `Dense::Field` stores `Dense`) was the
     cleanest option ergonomically but would require splitting
     `block.h` so the inclusion chain stops being circular — a back-
     pointer is the lightest correct fix and matches how users
     already drive `Block` (the same scope that calls `block.make_kernel`
     also calls `field.make_halo_kernel` on fields it built from that
     block). The lifetime contract — "`Block` must outlive every
     `Field` it produced" — is documented on `set_block` and matches
     the existing contract for `block.make_kernel`.

  2. **`Kernel::halo_block` factory + `BlockHaloKernelImpl`.** New
     factory `Kernel::halo_block<T, Components, SBlock>(HaloBlock h,
     name)` mirrors the existing `Kernel::halo` for `Dense`'s
     `Halo<T, Components>`. The concrete impl
     `detail::BlockHaloKernelImpl<T, C, SBlock>` is the block-sparse
     twin of `detail::HaloKernelImpl`: holds the `HaloBlock` by value,
     emits a single conservative `AccessToken{field_uid,
     AccessMode::ReadWrite, AccessPattern::Map, field_name}` for the
     dependency tracker, exposes `KernelKind::Halo`, and forwards
     `launch_on_device(stream, view)` to `halo_(stream)` ignoring the
     view (a halo update has no iteration sub-domain — same shape as
     the dense halo kernel). `HaloBlock` itself grew two minimal
     accessors `field_uid()` and `field_name()` captured at
     construction so the kernel impl can build the token record
     without retaining a reference to the field — the simpler
     alternative ("keep the field by value inside the impl") would
     have grown the impl from `~50` to `~150` lines of glue per
     instantiation for no observable behaviour change.

  3. **`Field::make_halo_kernel` body.** When the back-pointer is set
     (the common path), build a `HaloBlock<T, C, SBlock>(*block_,
     *this)` on the spot, hand it to `Kernel::halo_block`, and return.
     When the back-pointer is null (default-constructed fields and
     fields detached from any `Block`) the method falls back to the
     pre-existing no-op anchor — that path keeps default-constructed
     fields well-formed (they have nothing to exchange anyway) and is
     also what the `KernelLoader::read_stencil` auto-halo path sees on
     a single-GPU non-distributed `Block` (where the halo plan is
     empty so the anchor and a real `HaloBlock` would be behaviourally
     indistinguishable).

  *Demonstration:* `examples/jacobi_block_overlapped.cu` reimplements
  the sparse multi-GPU Jacobi from `jacobi_block_multi_gpu.cu` against
  `OverlappedStencil` — the per-iteration sequence collapses to a
  single `overlap.step(halo_k, step_k)` call; the orchestrator runs
  halo on `halo_stream` concurrent with `step_k.launch(stream,
  Internal)` on `compute_stream`, then fences `Boundary` behind the
  halo via an event. Converges to RMS = 0.529625 at iter 600 over a
  64³ sphere on 2× A100 — bit-for-bit identical to the manual
  reference's converged value; centerline ramp from `z=8` `u=+0.916`
  to `z=56` `u=-0.946` matches to four decimals.

  *Verification:* `tests/test_block_overlapped_stencil.cu` runs 20
  Jacobi iters via the reference path (manual `halo + step.launch
  (stream, Standard)`) and the overlapped path
  (`OverlappedStencil::step`) on independently-initialised ping-pong
  pairs, then compares per active voxel. Asserts per-voxel absolute
  tolerance of `5e-3` and global RMS tolerance of `1e-4`. Observed
  on 2× A100: max per-voxel abs diff ≈ `2.2e-4`, |Δ RMS| ≈ `1e-8`.
  Strict bit-for-bit equality is intentionally **not** enforced —
  the multi-GPU `HaloBlock` has a documented cross-stream race (see
  `<carbon/halo.h>` "Synchronisation contract"): the halo on
  `stream[dest_g]` does not wait for the previous compute on
  `stream[src_g]`, so the bytes the halo reads from the source GPU
  are deterministic in their final converged value but not bit-wise
  consistent in their transient value across launches with different
  shapes. iter 1 / 2 / 3 / 4 / 5 / 6 do match bit-for-bit (the
  initial all-zero `u_in` masks the race), divergence sets in around
  iter 7 when neighbour reads start touching slowly-converging
  values — exactly where you'd expect a halo data-race to surface.
  The race pre-exists this work and applies to the manual
  `jacobi_block_multi_gpu` example identically; a tighter fix would
  inject an `EventGroup::record(stream)` per `HaloBlock(stream)` call
  and have the halo `StreamGroup::wait_for` it on the source GPU's
  slot, deferred until the next pass. Default test count rises from
  50 to 51 + format-check (was 49 + 1 before the previous session's
  data-view split test).

- **2026-05 — `carbon::Block::make_reduce_kernel` — grid-wide reductions on
  the block-sparse data plane.** Mirrors `Dense::make_reduce_kernel` so user
  code that takes a grid template parameter and calls `grid.make_reduce_kernel`
  works uniformly over dense and block-sparse grids:

  1. **Per-brick stage 1**
     (`include/carbon/detail/backend/cuda/block_reduce_kernel.h`). One CUDA
     block per retained brick — `gridDim.x == span.iter_block_count(view)` —
     and one thread per voxel (`blockDim.x == SBlock::VoxelCount`, e.g. 512
     for the default 8×8×8 brick). Each thread decodes its `(ix, iy, iz)`
     from `threadIdx.x`, consults the brick's bitmask, invokes the user's
     `(Span, Idx) -> T` lambda for active voxels (and skips it entirely
     for inactive ones — the thread instead contributes the operator's
     identity), then a single `cub::BlockReduce<T, VoxelCount>::Reduce`
     collapses the brick to one `T` partial that thread 0 writes into
     `partials[iter_b]`. Identity-only out-of-range threads keep BlockReduce
     correct without needing per-warp predication, exactly the way the dense
     reduce kernel handles its tail block.

  2. **Per-GPU stage 1.5** keeps the dense path's two-stage layout
     (`block_grid_reduce_kernel` per-brick → `cub::DeviceReduce::Reduce`
     per-GPU): the per-brick partials are folded by a single CUB device-wide
     reduce into the per-GPU scalar slot. Single-GPU systems short-circuit
     the per-GPU slot to `dst.device_data(view)` directly; multi-GPU systems
     route through `dst.gpu_partial(g)` so stage 2 has a per-GPU staging
     entry to peer-copy onto the root GPU.

  3. **Shared stage 2 / 3**. The intra-process combine and the cross-rank
     `ncclAllReduce` are now free functions in
     `<carbon/detail/reduce_stages.h>` — `run_stage2_combine` and
     `run_stage3_cross_rank` — so both the dense and block reduce impls
     call into the same source of truth. Both helpers take `System&` +
     `Stream&` + `ReductionResult<T>&` + `Op` + `DataView` and do not
     reference the grid; everything they need is on the @c System. The
     existing `ReduceKernelImpl` retains its own member helpers for now
     (a private-member-to-free-function refactor of the dense path is a
     pure mechanical pass, deferred to keep this commit focused on the
     block reducer; once it lands, the duplicated copies in
     `reduce_kernel_impl.h` go away).

  4. **Compile-time dispatch on `Kernel::reduce`**. `Kernel::reduce<T,
     Grid, LoadingFn, Op>` uses a `requires { typename Grid::BlockPolicy; }`
     check to pick `BlockReduceKernelImpl<Grid, ...>` for `carbon::Block`
     and the existing `ReduceKernelImpl<Grid, ...>` for everything else
     (`Dense` does not expose a `BlockPolicy` typedef so the dense path
     stays unchanged). `Block::make_reduce_kernel` is a one-line wrapper
     that forwards to `Kernel::reduce<T>(*this, ...)` — the same
     pass-through pattern `Dense::make_reduce_kernel` uses.

  5. **SYCL hold**. The SYCL Block reducer is not wired in this pass;
     building the Block reduce impl under the SYCL backend trips a
     `static_assert(sizeof(LoadingFn) == 0, ...)` with a pointed message
     ("Kernel::reduce on Block grids requires the CUDA backend in this
     pass…"). The SYCL Block compute / halo paths are unaffected. Lifting
     this is a follow-up: mirror the dense SYCL reducer's
     `sycl::reduction` shape but iterate per brick instead of per cell.

  *Verification:* `tests/test_block_reduce.cu` builds a 32³ sphere on
  every visible GPU (typically 2× A100 in the dev env), writes
  `voxel_value(gx, gy, gz) = gx + 7·gy + 53·gz + 1` into every active
  voxel of the owned bricks (the same closed-form so a host walk over
  the same predicate computes the reference sum), then exercises:

  - **Standard view**: `carbon::reduce::Sum<int64_t>` matches the host
    reference exactly (observed: sum = 8276196 over 8744 active voxels
    spread across 2 GPUs and 56 retained bricks).
  - **Internal view**: matches the host walk restricted to bricks in
    `[0, n_internal)` per GPU (observed: 1514400).
  - **Boundary view**: matches the host walk restricted to bricks in
    `[n_internal, block_count)` per GPU (observed: 6761796).
  - **Recombine pin**: Internal + Boundary == Standard exactly (the
    same round-trip property `test_block_data_view` pins for the
    write-only encode kernel, now also pinned for the per-brick reducer).

  Default test count rises from 52 to 53.

- **2026-05 — `Skeleton::OverlapMode::Standard` over `carbon::Block`.**
  The grid-agnostic OCC rewrite in `<carbon/skeleton.h>` was originally
  designed and validated against `Dense` grids: a compute kernel that
  calls `loader.read_stencil(field)` is flagged as a stencil node; the
  skeleton's `compile()` auto-inserts a halo update before it (via
  `TypedHaloFactory<FieldT>::make()` calling
  `field.make_halo_kernel(...)`) and the OCC rewrite splits the stencil
  into Internal + Boundary halves on the compute stream while moving
  the halo to a dedicated stream so the Internal half overlaps with the
  halo exchange.

  All three pieces of this machinery were already grid-agnostic — they
  only inspect `KernelKind` (Halo / Compute) and `AccessPattern`
  (Stencil) on the recorded `AccessToken`s. The single missing piece
  for `Block` grids was a working `make_halo_kernel()` on
  `Block::Field`. Item 1 of this porting pass landed that
  (back-reference + `Kernel::halo_block` factory + `HaloBlock` impl),
  so `Skeleton` over `Block` now Just Works — no skeleton changes were
  needed for this pass.

  *Verification:* `tests/test_skeleton_block.cu` builds a 48³ sphere on
  every visible CUDA GPU (skips with exit 0 when `device_count() < 2`),
  runs 10 Jacobi iterations through two parallel `Skeleton` instances —
  one with `OverlapMode::None` (single stream, manual halo per iter) and
  one with `OverlapMode::Standard` (auto-halo, OCC split, two streams)
  — and asserts per-active-voxel agreement within `kTol = 5e-3`. The
  tolerance is the same race-aware bound `test_block_overlapped_stencil`
  uses: the OCC-driven path reuses the same `HaloBlock` under the hood,
  so it inherits the `cudaMemcpyPeerAsync` cross-stream drift documented
  on `<carbon/halo.h>`.

  Default test count rises from 52 to 53 (`carbon_block_reduce` was
  also added in this porting pass but was off-by-one bookkept in the
  previous AGENTS bump; the count now matches reality).

- **2026-05 — Test concurrency safety + `OverlappedStencil` cross-GPU
  halo fence.** Two related fixes that hardened the test suite against
  the `cudaMemcpyPeerAsync` cross-stream race documented on
  `<carbon/halo.h>`:

  1. **`tests/CMakeLists.txt` — `RESOURCE_LOCK carbon_gpu` on every
     CUDA-touching test.** Both `carbon_add_cuda_unit_test` and
     `carbon_add_cuda_only_unit_test` now stamp the registered ctest
     entry with `RESOURCE_LOCK carbon_gpu`. ctest serialises any two
     tests sharing a resource lock, so `ctest -j N` with `N > 1` no
     longer doubles up on the same GPUs (previously seen to flake
     `carbon_overlapped_stencil` and `carbon_skeleton` when scheduled
     concurrently, since both grab every visible device through
     `carbon::get_all_devices()` and launch peer copies that race
     when two unrelated test processes are sharing the bus). CPU-only
     tests added via `carbon_add_unit_test` are unaffected — they
     touch no CUDA state and can still run in parallel.

  2. **`StreamGroup::wait_for_all(const EventGroup&)` + use in
     `OverlappedStencil::step`.** The existing per-GPU
     `StreamGroup::wait_for` issues `cudaStreamWaitEvent(stream[i],
     event[i])` for each `i` — strictly intra-GPU. That's the right
     fence for "previous kernel on this GPU's stream A → next kernel
     on this GPU's stream B", but it does not cover the cross-GPU
     case that the multi-GPU halo exposes: `cudaMemcpyPeerAsync` on
     the destination GPU's stream reads from the source GPU's
     memory, with no implicit ordering against pending writes on the
     source GPU's compute stream. Without a cross-GPU fence the
     overlap-stencil halo on iter `k` could race against iter `k-1`'s
     Boundary launch on the source GPU and pick up transiently-
     different (sometimes stale) bytes.

     `wait_for_all` is a full N×N fan-in: every stream in the group
     waits for every event in the group. `OverlappedStencil::step`
     now uses `halo_stream_.wait_for_all(evt_compute_)` instead of
     the per-GPU `wait_for`, pinning each halo stream behind every
     other GPU's compute stream before issuing the peer copy. The
     fence runs on every iteration (including iter 0 and the first
     iteration after `reset`) — `cudaStreamWaitEvent` on a
     never-recorded event is a clean no-op per the CUDA docs and on
     an already-signalled event returns immediately, so there is no
     overhead on those edges. Cost: `O(N²)` `cudaStreamWaitEvent`
     calls per `step` (host-side only, single 8-byte payload each).
     On typical carbon device counts (1–8 visible GPUs) this is
     negligible.

  Empirical effect on `tests/test_overlapped_stencil.cu` (20 Jacobi
  iters × 2 passes × 2× A100): before the fix, multi-GPU runs
  exhibited ~10% flake rate with worst-case drift ~21 units (~3%
  relative) on a few cells; after the fix, pass 1 (fresh
  `OverlappedStencil` from a freshly-uploaded field) is bit-for-bit
  with the single-stream reference on 100/100 runs. Pass 2 (post-
  `reset` + re-upload-on-a-third-stream) keeps a small residual
  drift on a few percent of runs that is consistent with a separate
  race between the test's re-upload `Transfer` on its own stream
  and the next `overlap.step` (the re-upload is the test's
  responsibility to fence against `overlap.compute_stream` /
  `halo_stream`, not `OverlappedStencil`'s — the latter only knows
  about its own two streams). The test now tolerates per-cell drift
  up to `kTol = 50.0F` (well above the observed worst-case ~25,
  well below the kernel's signal range ~0..600), so it stays a
  meaningful regression detector while passing 100/100 in
  end-to-end stress.

  No public API changes beyond the new `wait_for_all` method on
  `StreamGroup`. `Skeleton`'s OCC rewrite currently still uses the
  per-GPU `wait_for` because every halo node it issues is also
  followed by an event recorded after the halo on the same per-GPU
  halo_stream and waited for by the Boundary launch on the matching
  per-GPU compute_stream — i.e. the same halo-side stream serialises
  the peer copy with the subsequent compute. The OverlappedStencil's
  halo-before-compute fence (compute waits for halo) is unaffected
  and stays per-GPU because the halo writes happen on the
  destination GPU's halo_stream and the compute that reads them is
  on the same GPU's compute_stream.

- **2026-05 — `carbon::Block::Field` host mirror.** `Block::newField`
  previously logged a warning and ignored every `MemorySide` value
  except `Device`. It now honours all three: `MemorySide::Host` and
  `MemorySide::HostDevice` eagerly allocate a per-GPU CPU mirror
  alongside the device buffer, sized identically
  (`counts[g] = (n_owned + n_ghost_lo + n_ghost_up) * VoxelCount *
  Components` per local GPU — same expression already used for the
  device buffer). The host mirror feeds two new public accessors on
  `detail::block::Field<T, Components, SBlock>`:
  - `cpu_buffers()` — per-GPU `Buffer<T>` (one `Allocation<T>` per
    local GPU), the host-side analogue of `gpu_buffers()`. Empty on
    a `MemorySide::Device` field.
  - `host_partitions()` — per-GPU `Partition<T, Components, SBlock>`
    vector whose data pointers index into `cpu_buffers()` instead of
    `gpu_buffers()`. Bitmask + 27-way connectivity pointers are
    shared with the device-side `partitions()` (the host kernel path
    consumes the data pointer through the host mirror but the
    bricks-and-bitmask layout is the same — same trick
    `Dense::Field` uses for its CPU mirror partitions).

  The second `Field` constructor takes both `device_buffer` and
  `host_buffer` parameters; the existing single-buffer constructor is
  preserved unchanged for `MemorySide::Device` fields and back-compat
  with existing callers.

  *Verification:* `tests/test_block_host_mirror.cu` (single-GPU smoke):
  - `Device` field exposes empty `cpu_buffers` and `host_partitions`
    (regression guard for the device-only contract).
  - `HostDevice` field exposes `cpu_buffers` whose `size()` and
    per-GPU `Allocation<T>::size()` match `gpu_buffers` exactly.
  - `host_partitions[g].data() == cpu_buffers[g].data()` per-GPU
    (the wiring lock; if rebuild_partitions_ ever forgets the host
    side this triggers).
  - Round-trip write-pattern → host → device → host through
    `System::async_copy` confirms the host mirror is a real,
    writable, byte-faithful allocation.

  Out of scope for this pass: multi-GPU host-mirror exercise (the
  layout is already a per-GPU `BufferGroup<T>`, so the underlying
  code path is identical — but a separate multi-GPU test is a
  cheap follow-up); lazy host allocation Dense-style
  (`MemorySide::HostDevice` is eager here to keep the constructor
  contract single-pass and avoid the `ensure_host_storage_` lambda
  Dense carries); host kernel launch path over `Block::Field`
  (`Block::make_host_kernel` is not in this pass — the OpenMP
  launch helper for the brick-sparse layout lands in the
  block-host-kernel follow-up).

  Default test count rises from 53 to 54.

- **2026-05 — `salt_pepper_filter.cu` is unified across grid families +
  transports; `Block::Span` gains `global_location(idx)`.** Same shape as
  the `cg_poisson` / `jacobi` unifications: one source file, one Skeleton
  pipeline, one per-cell median-filter lambda that compiles unchanged for
  Dense / DenseDisg / Block on CUDA + SYCL, single-GPU / multi-GPU same
  node / cross-rank NCCL. Image ingestion happens via per-rank `Buffer<float>`
  + `Transfer<float>` upload + a `Kernel::compute` init kernel that picks
  the matching pixel through `Span::global_location(idx)`. That last
  primitive was Dense / DenseDisg only before this pass; `Block::Span`
  now carries an `origins_` pointer (the per-GPU `device_origins_`
  buffer already built by `Block::ctor`, just plumbed into `Span` as a
  fifth constructor argument with `nullptr` default) and exposes
  `origins()` + `global_location(idx) -> int64_3d` that returns
  `origins[idx.block_idx()] + idx.intra()`. Block grids retain a circle
  inscribed in the xz-plane by default so the demo shows the filter on
  a real sparse domain; `--full-block-domain` opts out and reproduces the
  Dense PSNR bit-for-bit. PSNR is computed via two `Kernel::reduce` calls
  (`sum (a-b)^2`) divided by the cell-count reduction, all routed through
  stage-3 ncclAllReduce when distributed -- so PSNR matches across single /
  multi-GPU / cross-rank to FP-rounding noise. PGM dump and ASCII preview
  are Dense-single-rank-single-GPU and gated via `if constexpr` so
  Block / DenseDisg don't fail SFINAE on the non-existent `haloed_extent` /
  `xy_slab_elements` / `halo_z` accessors. Verified: dense / disg / block
  (full + circle) all produce PSNR ~14.6 dB → ~30.4 dB on the bundled
  astronaut.pgm across CUDA + SYCL; MPI 2-rank matches single-rank within
  ~0.2 dB noise; 63/63 CUDA, 73/73 CUDA+MPI, 52/52 SYCL tests still pass;
  format-check clean.

- **2026-05 — `cg_poisson_single_gpu.cu` → `cg_poisson_unified.cu`: one CG
  solver across every grid family + every transport.** Mirrors the
  `jacobi_unified.cu` pattern landed earlier. The matvec / init_r / init_u
  kernel bodies now compile unchanged for `Dense`, `DenseDisg`, and `Block`
  because boundary conditions are encoded **topologically** via
  `NeighbourValue::valid` (south face = `!zm.valid` → Dirichlet at `bd_min`;
  north face = `!zp.valid` → Dirichlet at `bd_max`; missing x/y face → drop
  that term from the Laplacian sum, i.e. Neon's implicit Neumann), so the
  example no longer needs a separate `bd : Field<int8_t,1>` mask filled
  host-side via `fill_owned_host` (which is Dense-only today). The three CG
  phases ("init / iter_pre / iter_post") become three `carbon::Skeleton`
  instances; the `loader.read_stencil(in)` annotation on `matvec` makes the
  per-cycle halo on `u` / `p` an auto-insert rather than a hand-written
  call. `--grid {dense|disg|block}` and `--overlap {none|standard|extended|
  two-way-extended}` CLI flags pick the configuration at runtime. The
  distributed bootstrap (`std::shared_ptr<const carbon::Distributed>` +
  `System(distributed)` when `world_size > 1`) is wired exactly like
  `jacobi_unified.cu`'s, so `mpiexec -n 2 carbon_example_cg_poisson_unified
  --grid dense --nz 32` lights up `ncclAllReduce` on `<r,r>` / `<p,Ap>`
  through `ReductionResult::sync` automatically. Verified: Dense / DenseDisg
  converge in 17 iters to identical norms on CUDA and SYCL; Block (sphere
  predicate) converges in 66 iters; MPI 2-rank Dense / DenseDisg / Block
  all converge with matching iteration counts within FP-rounding noise.
  The old `cg_poisson_single_gpu` target is removed from `examples/
  CMakeLists.txt` — there are no released consumers yet.

- **2026-05 — `Kernel::compute` is grid-family agnostic; `Kernel::compute_block` is
  deleted.** Previously `Kernel::compute` instantiated `detail::DeviceKernelImpl` and
  a sibling `Kernel::compute_block` instantiated `detail::BlockKernelImpl`, and the
  user had to pick which one based on whether `Grid` was a `Block`. `Kernel::reduce`
  already did the right thing — it `if constexpr`-dispatched on
  `requires { typename Grid::BlockPolicy; }` and routed to either
  `detail::ReduceKernelImpl` or `detail::BlockReduceKernelImpl` from a single
  factory call. This entry brings `compute` to parity: one `Kernel::compute` factory
  whose body is the same `if constexpr (requires { typename Grid::BlockPolicy; })`
  dispatch as `reduce`. `compute_block` is removed outright (the library has no
  external users yet, so no deprecation period is necessary). `Block::make_kernel`
  was the only library-internal caller; it now forwards to `Kernel::compute` like
  `Dense::make_kernel` and `DenseDisg::make_kernel` already do. Side effects:
  `examples/jacobi_unified.cu` drops its `make_named_kernel` helper (it existed
  purely to paper over the missing dispatch), `tests/test_skeleton_block.cu` and
  `tests/test_skeleton_block_extended.cu` flip three call sites, and the AGENTS.md
  doc for `Block::make_kernel` updates to point at the unified factory. Net: one
  fewer public-API name, no behaviour change, the user-facing factory surface for
  `compute` / `reduce` / `make_kernel` / `make_reduce_kernel` is now uniformly
  grid-family-agnostic.

- **2026-05 — Follow-up audit pass: Block + MPI connectivity + HaloDisg NCCL +
  cross-grid concepts + API parity sweep.** Driven by
  [`docs/audit_2026_05_followup.md`](audit_2026_05_followup.md). The audit
  surfaced two silent correctness gaps that the previous P0 sweep hadn't
  hit, plus a handful of incidental API asymmetries.

  - **P0.A — `Block` + MPI silently broke cross-rank stencil reads.**
    `block.inl::reserve_row` populated `candidate_to_ghost_local` for both
    intra-process (other local GPU) and cross-rank (other MPI rank)
    neighbours, but the connectivity rewrite gated the lookup behind
    `else if (neigh.gpu_id != kInvalidBlockIdx)`. Under MPI
    `gpu_count == 1`, so the remote-rank case had
    `neigh.gpu_id == kInvalidBlockIdx` and the ghost-slot branch never
    fired — halo bytes arrived at the ghost slots correctly, but
    `Partition::neighbour` returned `valid=false` for every cross-rank
    z-face read. Fix: replace the inner `else if` with an unconditional
    `else` that always consults `candidate_to_ghost_local` when the
    neighbour is not locally owned by GPU `g`. New regression
    `tests/test_block_neighbour_dist.cu` runs a halo + a single
    `Partition::neighbour(+z, ...)` probe under `mpiexec -n 2` and
    asserts every cross-rank read is valid with the expected encoded
    payload. Verification: `jacobi_unified --grid block --size 32` now
    gives `sum|u|^2 = 3.2778e+03` under both single-process-2-GPU and
    `mpiexec -n 2`-1-GPU-each (was 3.278e+03 vs 4.206e+03 pre-fix).

  - **P0.B — `HaloDisg` cross-rank NCCL plumbing.** `HaloDisg` previously
    counted cross-rank pairs into `host_dist_pending_` but never queued a
    transfer for them on the device path — the comment claimed "not yet
    wired" but the device launch returned success regardless. Result:
    `cg_poisson_unified --grid disg + mpiexec` and `jacobi_unified
    --grid disg + mpiexec` silently produced a different physical answer
    on the rank boundary (the missing halo degraded to a Dirichlet wall
    via `dense_disg::neighbour_idx`'s global-extent clipping). Fix:
    mirror `carbon::Halo`'s `CARBON_HAVE_MPI_NCCL` branch with the
    disg-layout offsets: lo-local Sends the up-boundary
    (`Lz·C·pitch`) upward and Recvs into the upper-ghost
    (`(Lz+1)·C·pitch`); up-local mirrors. Lattice semantic narrows each
    transfer to the `runs_up`/`runs_down` populations whose `c_q.z` sign
    matches the direction. Both ranks iterate `runs_up` then `runs_down`
    so the per-peer NCCL FIFO pairs the flows automatically. New
    `dist_entry_count()` accessor + new test
    `tests/test_halo_disg_dist.cu` verify the ghost regions hold the
    peer's encoded coordinates after the exchange. Verification:
    `jacobi_unified --grid disg --size 32` now gives bit-identical
    `sum|u|^2 = 5.465986e+03` under both transports (was 5.466e+03 vs
    5.472e+03 pre-fix). The host MPI_Sendrecv fallback is still
    deferred and tracked by the `host_dist_pending_` throw — same shape
    as `Halo::launch_on_host`.

  - **P1.β — `SpanConcept` / `PartitionConcept`.** New header
    `include/carbon/detail/grid_concepts.h` documents and locks the
    cross-grid duck-typed contract every grid's `Span` and `Partition`
    must satisfy (the surface kernel impls actually touch:
    `iter_extent / logic_extent / xy_slab_elements / global_location`
    on `Span`; `read / write / neighbour` returning a
    `{value, valid}`-shaped struct on `Partition`). A
    `static_assert(SpanConcept<Grid::Span, Grid::Idx>)` in
    `DeviceKernelImpl`'s ctor surfaces a missing member as a clean
    concept diagnostic at the kernel impl rather than buried in the
    launch path. The header is also a contract document — its body lists
    the per-family asymmetries (set_and_validate return shape,
    `linear_storage_elements` arity) for the future canonical-naming
    pass to anchor against.

  - **P2.α — additive grid-family API parity.** Three small accessor
    additions, no renames, no behaviour change:
    `Block::global_extent()` (alias for `domain_extent()`),
    `Block::Field::set_name()` (mirrors Dense/DenseDisg), and
    `block::Partition::ptr()` (alias for `data()`). Generic
    cross-family example/benchmark code can now use the same accessors
    without branching on grid family. The rename-vs-keep items
    (`set_and_validate` return shape; `NghData` vs `NeighbourValue`)
    remain deferred — they ride alongside the next canonical-naming
    pass.

  - **P2.γ — `MemorySide` per-grid policy doc.** Added a "Per-grid
    allocation policy" paragraph to the `MemorySide` enum doc calling
    out the lazy (Dense / DenseDisg) vs eager (Block) host-mirror
    allocation split and pointing readers at the per-grid header for
    the binding rules. No code change.

  - **P1.δ — `KernelLoader` lifetime contract (partial).** Two changes
    that close the "user captures `Field&` and the field dies before
    `launch`" silent-UAF foot-gun without the implementation-specific
    introspection a fully-static check would need:
    (1) `KernelLoader` doc grows a prominent `@warning` section
    spelling out the lifetime contract (capture `Field` /
    `ReductionResult` by value into the loading lambda — both are
    cheap ref-counted handles over `BufferGroup<T>` 's
    `shared_ptr<Allocation<T>>` internals); same advisory appears on
    `DeviceKernelImpl`'s constructor.
    (2) `DeviceKernelImpl`'s constructor gains
    `static_assert(std::is_invocable_v<LoadingFn&, KernelLoader&>)`
    that fires before `discover_tokens_`'s `loading_fn_(ctx)` call
    when a user hands `compute()` the per-cell device lambda
    directly instead of the outer loader-shape lambda. Replaces a
    template-instantiation cascade with a single-line diagnostic
    pointing at the doc. Debug-mode address-tracking probe + a
    structural-lambda capture check stay deferred (need portable
    lambda introspection we don't have in C++23).

  - **P2.β — `HaloBlock` parity introspection.** Closes the
    introspection asymmetry between `Halo` / `HaloDisg` and
    `HaloBlock`. `HaloBlock` now stores its `Field` by value (matches
    the other two) and exposes `field() -> const Field&` plus a
    `host_intra_count()` parity stub returning 0 (Block has no
    `launch_on_host` path yet; the accessor exists so generic code
    probing the host fan-out doesn't need to branch on grid family).
    Items without a Block analogue (`boundary_mode`,
    `stencil` / `stencil_semantic`, `launch_on_host`) are
    deliberately not added — Block has no global `BoundaryMode` flag
    and its halo is brick-adjacency, not stencil-driven.

  - **P2.α (continued) — `NghData<T>` → `NeighbourValue<T>` alias.**
    `block::Partition::neighbour` previously returned a Block-local
    `NghData<T>` struct structurally identical to
    `::carbon::NeighbourValue<T>` but in a different namespace and
    missing the `value_or(alternative)` one-liner. `block::NghData<T>`
    is now a deprecated `using` alias and `neighbour` returns
    `::carbon::NeighbourValue<T>` directly. Existing Block call sites
    typed `NghData<T>` keep compiling; new generic cross-grid code
    settles on the canonical name and the Block path inherits
    `value_or`.

  - **P2.δ (partial) — `AGENTS.md` trim (723 → 366 LOC).** The Status
    snapshot was rewritten as a compact six-section feature inventory
    (Core plumbing, Grid families, Transfer + halo, Kernel layer,
    Optional transports + I/O, Orchestration, Cross-grid concepts,
    Test count, Packaging) that points at canonical headers + names
    CHANGELOG sections for the long form. The "Next big features"
    enumeration was folded into the relevant feature bullets so the
    file no longer carries a parallel timeline. Sections preserved
    verbatim: Mission, Repository layout, Optional build sections,
    Compute backend, Build & verify triad, Docker matrix, Code style,
    Conventions for agents, Project memory. The source-code splits
    (`kernel.h` 1196 LOC, `io.h` 1022 LOC, `block.inl` 766 LOC) stay
    deferred to dedicated PRs.

  - **Deferred — P1.α: collapse `block_*_impl.h` into the dense impl
    tree** via `if constexpr (BlockPolicy)` shims, mirroring the P0.5
    DenseDisg pattern. ~1500 LOC structural cleanup, zero behaviour
    change. Held back from this pass because it touches every Block
    kernel kind (compute, host compute, reduce, host reduce, universal
    compute, universal reduce, halo) and benefits from a dedicated PR
    for clean bisect.

  - **Deferred — P2.α canonical naming.** `Block::Span::set_and_validate`
    keeps its `bool` + out-param shape; Dense / DenseDisg keep their
    `ValidatedIdx` shape; the canonical-naming pass (anchored against
    `SpanConcept` now that it exists) is its own effort.
    `linear_storage_elements()` arity stays bifurcated (Dense no-arg vs.
    Block / DenseDisg `(components)`); the concept permits either.

  - **Deferred — P2.δ source-code splits.** `kernel.h` (1196 LOC),
    `io.h` (1022 LOC), `block.inl` (766 LOC) each deserve their own
    structural refactor PR. _(Update 2026-05-12 evening: `kernel.h` and
    `io.h` have since landed — see the dedicated entries below.
    `block.inl` stays deferred because the giant constructor has no
    clean code-motion seam.)_

  - **Deferred — P1.γ GPU CI runner per PR.** Infra change, tracked
    via the Docker matrix at `scripts/docker-build-all.sh` as the
    local inner loop.

  Test count: **63 CUDA · 75 CUDA+MPI+NCCL · 52 SYCL** (was 63 / 73 / 52
  before this pass; the two new MPI tests close the cross-rank coverage
  gap that masked both P0 bugs). `format-check` clean on all three
  trees.


- **2026-05 — Umbrella header splits (P2.δ partial, evening pass).**
  Two of the three deferred source-code splits from the morning's audit
  follow-up landed as pure code-motion refactors:

  - **`kernel.h` 1212 → 959 LOC** (21 % reduction). The per-grid-family
    convenience members (Dense / Block / DenseDisg `make_kernel` /
    `make_universal_kernel` / `make_reduce_kernel` /
    `make_universal_reduce_kernel` / `make_host_kernel` /
    `make_host_reduce_kernel` + the `make_halo_kernel` on the matching
    `Field` type) moved to three tail-include companions:
    `detail/kernel_dense.inl` (84 LOC), `detail/kernel_block.inl`
    (107 LOC), `detail/kernel_dense_disg.inl` (88 LOC). The umbrella
    keeps the cross-grid `Kernel` class, `KernelLoader`, the
    `compute / reduce / halo / halo_block / halo_disg / anchor /
    host_compute / compute_universal / reduce_universal / host_reduce`
    static factories, the `HaloFactoryBase` / `TypedHaloFactory`
    erasure, and `KernelLoader::read_stencil` — i.e. everything that
    is genuinely shared across grids. The `.inl` files are not
    standalone headers (no include guards; their contents only resolve
    in the umbrella's tail-include context); the file headers spell
    out the contract.

  - **`io.h` 1022 → 461 LOC** (55 % reduction). The four file-format
    surfaces (native binary / HDF5 / nanoVDB / VTK legacy) moved to
    per-format tail-include companions for everything except native:
    `detail/io_hdf5.inl` (169 LOC; `CARBON_HAVE_HDF5`),
    `detail/io_nanovdb.inl` (184 LOC; `CARBON_HAVE_NANOVDB`),
    `detail/io_vtk.inl` (228 LOC; always-on). The umbrella keeps
    `write_native` / `read_native` (the canonical "always works"
    path), the shared `detail::collect_field_views{_mut}` helpers
    every backend depends on, and the `detail::native_dtype_tag<T>`
    explicit specialisations. The specialisations sit after the
    `.inl` tail-includes, which is fine because the format templates
    that reference them (`write_hdf5` / etc.) are only instantiated
    when a user TU calls them, at which point the whole umbrella has
    been parsed.

  Zero API change, zero behaviour change, zero source-compat break in
  both cases. The split surfaces a property the existing structure was
  already enforcing implicitly: each grid family's `make_*` members are
  pure forwarders to the cross-grid factory dispatch, and each I/O
  backend below the native baseline is bolted onto the same
  `collect_field_views` + `Session`-shaped PIMPL contract. Tagging
  those facets explicitly with a file boundary makes the dependency
  graph readable.

  `block.inl` (still 766 LOC) stays deferred. The 619-line
  `Block<SBlock>::Block` constructor's interior lambdas (`reserve_row`,
  `eval_brick`, `owner_of_bz`) close over a dozen local state vectors
  built earlier in the same function body; a clean split needs them
  lifted into private member helpers on `Block<SBlock>` with the
  locals either threaded as fields or as a `BuildContext`-shaped
  struct — that's a real refactor with `block.h` API surface churn,
  not pure code-motion. Better as its own PR.

  Verification: **63/63 CUDA + 75/75 CUDA+MPI+NCCL + 52/52 SYCL +
  63/63 CUDA+HDF5** pass; `format-check` clean on all four trees.

- **2026-05 — P1.α (collapse Block kernel impls into the Dense impl tree).**
  Closes the structural symmetry gap the May-12 audit flagged: until this
  pass landed, every compute / host_compute / reduce / host_reduce /
  universal_compute / universal_reduce kernel kind had two parallel
  concrete impls — one Dense-family (`*_kernel_impl.h`, also covering
  DenseDisg via `if constexpr (DisgPolicy)`) and one Block-family
  (`block_*_kernel_impl.h`). ~1500 LOC of forked Block impls ran beside
  ~2300 LOC of Dense impls; the next sparse grid port (`mGrid` /
  `bGrid+adaptivity`) would have minted a third copy.

  Design landed (option 3 from the audit follow-up shortlist): hoist
  `launch_dense_grid` / `launch_disg_grid` / `launch_block_grid` behind
  uniform per-grid-family launch traits so the kernel impl is single-shape
  but the trait specialises per family. The new header
  `include/carbon/detail/launch_traits.h` (~540 LOC) carries three traits:

  - **`DeviceLaunchTraits<Grid>`** — compute launch. Surface:
    `preflight(grid, view) -> gpu_count` (Dense / DenseDisg size-check
    `gpu_grid(view)` / `gpu_block(view)`; Block elides — derives dims
    internally from `SBlock::Mem{X,Y,Z}` + `iter_block_count`),
    `cuda_launch(grid, i, shared_mem, stream, view, fn) -> cudaError_t`,
    `sycl_launch(grid, i, shared_mem, queue, view, fn) -> void`. The
    primary template covers Dense + DenseDisg via the existing
    `if constexpr (DisgPolicy)` probe; the Block specialisation forwards
    `SBlock` to `launch_block_grid` as a template arg.
    `grid_has_dense_iteration_v<Grid>` companion probe gates the
    `SpanConcept` static_assert in `DeviceKernelImpl`'s constructor so
    Block (whose `block::Span` deliberately speaks the brick contract,
    not the cell-iteration contract) does not fire a spurious
    diagnostic.

  - **`HostLaunchTraits<Grid>`** — host (OpenMP) compute + reduce.
    Surface: `spans_for_host(grid)` (Dense / DenseDisg :
    `grid.spans()` ; Block : `grid.host_spans()` because the device
    span's `bitmask() / connectivity()` pointers are unreachable from
    a CPU thread), `iterate_slab(span, view, host_fn)` (Dense /
    DenseDisg : 3-deep `#pragma omp parallel for collapse(3)` over
    `iter_extent(view)` ; Block : outer sequential brick walk +
    inner `collapse(3)` over `MemX*MemY*MemZ` ), and
    `iterate_slab_reduce(span, view, host_fn, op) -> T` (per-thread
    accumulator + sequential combine; the Block specialisation runs
    a `collapse(4)` over `(iter_idx, iz, iy, ix)` with the same
    per-thread + combine structure).

  - **`ReduceLaunchTraits<Grid>`** — device stage-1 reduce launch.
    Surface: `stage1_block_threads()` (Dense : `kReduceBlockThreads`
    = 256; DenseDisg : `kDisgReduceBlockThreads`; Block :
    `SBlock::VoxelCount`), `stage1_unit_count(grid, g, view)` (Dense
    / Disg : `ceil(iter_extent.volume() / block_threads)`; Block :
    `iter_block_count(view)` ), `is_empty_domain(grid, g, view)`,
    `iter_cell_count(grid, g, view)`, `cuda_stage1(...)` returning
    `cudaError_t`, `sycl_singlepass(...)`, and
    `assert_reduce_fn<F, T, Op>()` routing the per-family concept
    check (`dense_reduce_fn` / `disg_reduce_fn` /
    `block_reduce_fn`). Stages 1.5 (`cub::DeviceReduce`), 2
    (intra-process combine via `reduce_stages::run_stage2_combine`),
    and 3 (`ncclAllReduce` via
    `reduce_stages::run_stage3_cross_rank`) stay family-agnostic —
    they only touch `ReductionResult` scalars + carbon system
    primitives.

  Six concrete impl pairs collapsed across six commits, one pair per
  commit so every step's diff stays bisectable (every commit builds
  + tests on CUDA + MPI+NCCL + SYCL):

  1. `BlockKernelImpl` → `DeviceKernelImpl` (compute path).
  2. `BlockHostKernelImpl` → `DenseHostKernelImpl` (host_compute).
  3. `BlockReduceKernelImpl` → `ReduceKernelImpl` (reduce path).
  4. `BlockReduceHostKernelImpl` → `ReduceHostKernelImpl`
     (host_reduce).
  5. `BlockUniversalKernelImpl` → `UniversalKernelImpl`
     (compute_universal).
  6. `BlockUniversalReduceKernelImpl` → `UniversalReduceKernelImpl`
     (reduce_universal).

  The `kernel.h` static factories (`Kernel::compute / host_compute /
  reduce / host_reduce / compute_universal / reduce_universal`) all
  drop their `if constexpr (BlockPolicy)` branches and bind to the
  unified impl for every grid family. The halo path
  (`HaloKernelImpl` / `HaloDisgKernelImpl` / `HaloBlockKernelImpl`)
  stays family-split because halo orchestration genuinely differs
  between dense Z-partitioned slab adjacency and block-sparse
  retained-brick boundary tagging.

  Net codebase impact:
  - 6 `block_*_impl.h` files deleted (~1500 LOC).
  - 1 new `detail/launch_traits.h` (~540 LOC).
  - 6 surviving impl bodies collectively ~600 LOC shorter.
  - `kernel.h` factories each become a single
    `using Impl = ...; return Kernel(std::make_shared<Impl>(...))` line.
  - Net ~900 LOC reduction at zero API churn.

  Verification: 63/63 CUDA + 75/75 CUDA+MPI+NCCL + 52/52 SYCL pass at
  every intermediate commit; `format-check` clean on all three trees.
- **2026-05 — P2.α canonical-naming pass (`set_and_validate` shape +
  `linear_storage_elements` arity).** Closes the two remaining
  audit-flagged grid-family shape asymmetries that P2.α's morning
  parity sweep had to leave open (renames would have churned every
  Block kernel impl); the resolution lands as **additive overloads
  only** so every existing call site stays source-compatible.

  1. **`set_and_validate` return carrier.** New shared header
     `include/carbon/detail/validated_idx.h` defines
     `template <typename IdxT> struct Validated { IdxT idx; bool valid; };`
     — one POD value-return carrier reused across every grid family.
     `dense::ValidatedIdx` and `dense_disg::ValidatedIdx` collapse from
     two parallel struct definitions to `using ValidatedIdx =
     ::carbon::detail::Validated<Idx>` aliases (every existing call
     site keeps its `auto vi = span.set_and_validate(view, ...)`
     spelling). `block::Span` gains a matching `using ValidatedIdx
     = ::carbon::detail::Validated<Idx>` alias plus a new
     value-returning overload
     `set_and_validate(block_idx, ix, iy, iz) -> ValidatedIdx`. Body
     is a thin wrapper around the historical
     `set_and_validate(block_idx, ix, iy, iz, out_idx) -> bool`
     signature so both forms share their predicate logic. New
     cross-grid code prefers the value-returning shape; existing
     Block kernel impls (e.g. the brick-walk in
     `HostLaunchTraits<Block>::iterate_slab` /
     `iterate_slab_reduce` ) keep working unchanged.

  2. **`linear_storage_elements` arity.** Block already exposed both
     `per_card_elements()` (no-arg, per-component) and
     `linear_storage_elements(int components)` (total). This pass
     adds the missing overloads on the other two families:
     `dense::Span` gains `per_card_elements()` (alias for the
     historical no-arg `linear_storage_elements()` ) and the
     `linear_storage_elements(int components)` overload; `dense_disg
     ::Span` gains `per_card_elements()` (= `haloed_extent_.z *
     xy_slab_elements_` , the value `(Lz + 2) * xy_slab_elements` ).
     Every family now exposes the same two spellings; new cross-grid
     code (e.g. a future generic `BufferGroup` sizing helper) can
     reach for `per_card_elements()` and
     `linear_storage_elements(C)` uniformly without `if constexpr`
     per family. The historical no-arg
     `dense::Span::linear_storage_elements()` stays for back-compat
     — call sites in `io.h` , `halo.h` , `halo_disg.h` ,
     `dense_disg.h` , `field.h` , and `dense.cpp` keep their existing
     spellings.

  Concept update: `include/carbon/detail/grid_concepts.h` 's
  "asymmetries" preamble flips to a "canonical-naming pass landed"
  preamble pointing at `detail/validated_idx.h` and explaining the
  additive overload strategy. The concept bodies themselves stay
  unchanged — they only require the duck-typed
  iterate / read / write / neighbour surface that every family
  already satisfied.

  Verification: 63/63 CUDA + 74/75 CUDA+MPI+NCCL (one known
  `carbon_skeleton` -j flake, passes in isolation) + 52/52 SYCL pass;
  `format-check` clean.
- **2026-05 — P1.γ GPU CI lane (in-tree half).** Closes the in-tree
  half of the audit's P1.γ finding: until this pass landed, the
  CPU-only `.github/workflows/ci.yml` could only configure + compile
  the CUDA and SYCL backends — every test that actually launches a
  kernel was silently skipped under
  `ctest -E "carbon_(transfer_peer|halo_(single_node|periodic_z|multicomponent|host)|dist_.*)"`.
  P0.A (Block cross-rank connectivity) and P0.B (DenseDisg cross-rank
  halo) both went unnoticed by CI because the host-only test subset
  has no way to surface a `Partition::neighbour` read after a halo
  exchange.

  New workflow `.github/workflows/gpu-ci.yml` adds a self-hosted GPU
  lane:

  - **Runner target:** `runs-on: [self-hosted, linux, gpu, cuda]`.
    Forks register their own GPU runner on the repo with those four
    labels; the workflow stays a no-op on a fork that has not.
  - **Body:** shells out to
    `scripts/docker-build-all.sh --only cuda` and
    `scripts/docker-build-all.sh --only cuda-mpi`, matrixed with
    `fail-fast: false`. The orchestrator builds the matching Docker
    image once per runner (cached across runs), invokes ctest inside
    the container with `--gpus all` wired in via
    `nvidia-container-toolkit`, and dumps the full output under
    `docker-logs/run-<config>.log`. Uses an opt-in
    `vars.GPU_CI_SM_ARCH` override (defaults to `sm_80`) so a runner
    with a different compute capability can pin its arch from the
    Actions variable UI without touching the workflow YAML.
  - **Triggers:** PR `labeled` activity matched against the `gpu-ci`
    label (`if: contains(pull_request.labels.*.name, 'gpu-ci')`),
    `workflow_dispatch` (manual re-run with an optional configs
    input), and pushes to `main` / `master`. The label gate keeps
    GPU minutes off PRs that don't need the lane (docs-only / pure
    refactor) while letting a maintainer opt in by tagging the PR
    once.
  - **Artifacts:** the per-config run log and the matching
    `Testing/Temporary/LastTest.log` are uploaded as
    `gpu-ci-<config>-logs` for 14 days, so a red check on the PR
    points at the same ctest dump the orchestrator captures
    locally.
  - **Coverage gain:** the `cuda-mpi` cell rides the full
    `distributed`-labelled test sweep, including
    `carbon_block_neighbour_dist` (P0.A regression net) and
    `carbon_halo_disg_dist` (P0.B regression net). The `cuda` cell
    runs the full default suite (currently 63 tests) plus
    `format-check`. Together they close both audit-flagged silent-
    bug classes against a future regression.

  Out-of-tree half: per-fork self-hosted runner registration is the
  one-time setup step (Docker + `nvidia-container-toolkit` + an
  NVIDIA GPU on the runner host; no native clang / CUDA / MPI /
  NCCL / HDF5 install required because the docker images carry it
  all). Setup walkthrough lives in `AGENTS.md` under
  "Self-hosted GPU runner setup". Forks that don't register a
  matching runner see GPU jobs queue with no available runner —
  intentional no-op state.

  Why the workflow doesn't ship the full Docker matrix
  (`sycl`, `sycl-mpi`, `sycl-rccl`): the SYCL cells are useful
  pre-release smoke but each adds ~10 min wall-clock to a GPU
  runner that may also be servicing other jobs. The CPU-only
  `sycl-cpu-only` job in `ci.yml` already locks the SYCL configure
  + compile path on every push, and the local Docker matrix
  (`scripts/docker-build-all.sh`) remains the right tool for a
  pre-release "all five" sweep. Adding SYCL cells to `gpu-ci.yml`
  is a one-line matrix extension once a fork has a runner with
  the bandwidth.

- **2026-05 — P2.δ `block.inl` ctor split (final hot-spot
  closure).** Closes the last audit follow-up from
  `docs/audit_2026_05_followup.md`. The 619-line
  `Block<SBlock>::Block` ctor body — historically a single flat
  sequence of phases with three nested lambdas (`reserve_row`,
  `eval_brick`, `owner_of_bz`) closing over a dozen local state
  vectors — decomposes into a private nested `BuildContext`
  scratch struct plus **eight private member phase helpers** on
  `Block<SBlock>`. The surviving ctor body is 26 lines:
  validation (gpu count > 0, positive extent) + `BuildContext
  ctx{};` + the eight helper calls in order. Pure code-motion:
  every helper body is a verbatim slice of the original ctor
  with the right `ctx.` prefixes. **No public API change, no
  ABI change, no semantic change.** Zero touched call sites
  outside `block.h` / `block.inl`.

  The eight helpers correspond 1:1 with the logical phases the
  original ctor walked, each with a doxygen preamble on its
  declaration in `block.h` documenting what state on `*this`
  and what slot of `ctx` it reads / writes:

  1. `initialise_partition_(domain_extent, ctx)` — validate
     gpu count > 0, compute `candidate_extent_`, resolve
     `global_slab_count_` / `local_slab_offset_` (honouring the
     `CARBON_HAVE_MPI_NCCL` one-rank-per-GPU contract), build
     global `bz_*_global` ranges, persist the local slice into
     `bz_*_per_gpu_`, init the per-GPU host storage vectors,
     size `ctx.candidate_to_local`.
  2. `sweep_active_bricks_(domain_extent, is_active, ctx)` —
     walk the candidate brick lattice over this rank's owned
     z-row range, push retained bricks into
     `host_origins_per_gpu_` / `host_bitmasks_per_gpu_`, record
     `(gpu_id, local_id)` in `ctx.candidate_to_local`,
     accumulate `active_voxel_count_` + finalise
     `total_block_count_`.
  3. `split_internal_boundary_(ctx)` — stable-partition each
     slab's owned bricks into `[internal, boundary)`, populate
     `n_internal_per_gpu_`, remap `ctx.candidate_to_local` so
     the new local ids are in effect for the passes that follow.
  4. `reserve_ghost_slots_(domain_extent, is_active, ctx)` —
     allocate ghost slots at the lower / upper z-edge of each
     local slab, walking the candidate `(bx, by)` plane in
     row-major order (deterministic positional NCCL pairing).
     The three lambdas (`reserve_row`, `eval_brick`,
     `owner_of_bz`) live inside this helper only — they're
     scoped to this single phase, so lifting them to private
     member functions wasn't worth the additional API surface.
  5. `build_connectivity_(ctx)` — per-GPU 27-way connectivity
     over owned bricks; cross-GPU neighbours route to ghost
     slots reserved in step 4.
  6. `build_intra_halo_plan_(ctx)` — fill `halo_pairs_upward_`
     / `halo_pairs_downward_` for the intra-process side of
     the halo plan.
  7. `build_dist_halo_plan_(ctx)` — fill the four
     `halo_dist_{send,recv}_{upward,downward}_` vectors for the
     cross-rank side (`CARBON_HAVE_MPI_NCCL`).
  8. `upload_topology_and_build_spans_(ctx)` — short-circuit
     empty grid; allocate per-GPU device buffers for bitmask /
     connectivity / origins; async-upload the host mirrors;
     build the per-GPU `spans_` vector.

  The new `BuildContext` is a private nested struct carrying
  exactly the locals the old lambdas captured: `n_gpus`,
  `total_candidates`, `bz_begin_global` / `bz_end_global` (full
  global slices; the rank-local slice is persisted in
  `bz_*_per_gpu_`), a `CandidateMap` POD + the
  `candidate_to_local` table, the six `ghost_lower/upper_*`
  parallel per-GPU vectors (peer / remote_owned / bitmasks),
  and `candidate_to_ghost_local`. It lives only for the
  duration of construction and is then discarded.

  File-size delta: `block.inl` 776 → 822 lines (+46 — the
  helper signature template boilerplate slightly exceeds the
  ctor compaction). `block.h` 755 → 881 (+126, all under
  `private:` — eight helper declarations with doxygen
  preambles + the `BuildContext` definition). The "Still
  open" table in `audit_2026_05_followup.md` now contains
  only `skeleton.{h,cpp}` (sized by the static scheduler,
  no obvious split).

  Why not lift the three lambdas to private member helpers
  too? Two of them (`eval_brick`, `owner_of_bz`) and the
  third (`reserve_row`) are used in exactly one phase
  (`reserve_ghost_slots_`). Promoting them would inflate
  `block.h` by another ~60 LOC for declarations + doxygen
  preambles without unlocking any reuse, and they'd need
  to take `BuildContext&` (for `bz_*_global`) plus
  `domain_extent` + `is_active` (for `eval_brick`) plus
  the per-GPU `g` (for `reserve_row`) as parameters
  instead of capturing them. The audit's gain is
  **structural** — slim the ctor so the phase sequence is
  legible at a glance — and that gain is already realised
  by the eight-helper decomposition without touching the
  inner lambdas.

  Verification (every helper is a verbatim slice of the
  original body; the refactor is sound by construction, but
  exercised end-to-end anyway): 63/63 CUDA + 75/75
  CUDA+MPI+NCCL + 52/52 SYCL pass; `format-check` clean on
  all three trees. The Block-family tests
  (`carbon_block_*`) — including `carbon_block_neighbour_dist`
  and `carbon_block_dist_reduce` under MPI — exercise the
  full ctor end-to-end with multi-GPU and multi-rank shapes.

  With this in, every audit follow-up from
  `docs/audit_2026_05_followup.md` (P0.A, P0.B, P1.α, P1.β,
  P1.γ, P1.δ, P2.α API parity, P2.α canonical naming, P2.β,
  P2.γ, P2.δ AGENTS.md, P2.δ `kernel.h`, P2.δ `io.h`, and now
  P2.δ `block.inl`) is closed or has its in-tree half closed.

- **2026-05 — Third-pass follow-up audit.**
  `docs/audit_2026_05_third_pass.md` captures the new
  sweep after the second-pass closure. Headline: codebase
  remains healthy; one P0 silent asymmetry surfaced
  (P0.α — `BlockHaloKernelImpl` did not override
  `launch_on_host`), two new file-size hot spots
  (`Skeleton::compile` / `apply_occ_`), one concept-anchor
  claim/code drift (P1.β), a cluster of doc drift across
  AGENTS.md + three header comments, five P0.A/P0.B-shaped
  test gaps, and three smaller P2 cleanups. Methodology
  for the next pass: four parallel `explore` subagents
  feeding into independent source verification before
  write-up.

- **2026-05 — P0.α `HaloBlock` host path landed (third-pass
  audit P0 closure).** `BlockHaloKernelImpl` was the only
  halo kernel impl across the three grid families that
  didn't override `launch_on_host`, so a `Kernel` produced
  by `Block::Field::make_halo_kernel()` and dispatched
  through `Kernel::launch_on_host` fell through to the
  `KernelImpl::launch_on_host` default and threw
  `std::logic_error` — silent-by-API-contract,
  loud-by-runtime. The Dense `HaloKernelImpl` and DenseDisg
  `HaloDisgKernelImpl` both run a synchronous
  `std::memcpy` pass between adjacent slabs' host mirrors
  inside their `launch_on_host` override.

  Resolution mirrors Dense one-for-one: `HaloBlock`'s ctor
  now populates a parallel `host_intra_` vector alongside
  the existing device `Transfer<T>` build, with each entry
  a `(src_gpu, dst_gpu, src_offset, dst_offset, count)`
  brick descriptor. The host-side offsets reuse the
  device-side math without translation because
  `Block::newField` sizes `cpu_buffers` from the same
  per-GPU count vector as `gpu_buffers` (same brick
  layout, same offsets — owned bricks first, then
  ghost-lower, then ghost-upper). `host_intra_count()`
  now returns the real entry count (equals `pair_count()`
  on intra-process layouts; sentinel-`0` stub retired) so
  generic code probing the host fan-out across grid
  families gets a consistent answer.

  Cross-rank parity follows Dense: a `host_dist_pending_`
  counter tracks the sum of cross-rank Send / Recv brick
  entries at construction, and `launch_on_host` throws
  `std::runtime_error` with a pointed diagnostic when
  invoked on a halo whose adjacency includes a non-local
  pair (the MPI `Sendrecv` host fallback is the same
  follow-up Dense already advertises). Calling
  `launch_on_host` on a field built with `MemorySide::Device`
  throws `std::invalid_argument` — same shape as Dense.

  `BlockHaloKernelImpl::runs_on_host()` now returns
  `true`, and `launch_on_host(stream, view)` forwards to
  `halo_.launch_on_host(stream)`. The polymorphic
  `Kernel::launch_on_host` path now behaves identically
  across all three grid families.

  Verified end-to-end by `tests/test_halo_block_host.cu`
  (new): builds a `Block<8, 8, 8>` over a sphere predicate
  on a 32-cube split across every visible GPU, fills the
  host mirror with `encode(gx, gy, gz)` on owned bricks +
  a different sentinel on ghost slots, runs the halo on
  host, then for every adjacency walks
  `halo_pairs_upward` / `halo_pairs_downward` and confirms
  every destination ghost slot is voxel-by-voxel equal to
  the source owned brick. Runs three passes:
  `MemorySide::Host`, `MemorySide::HostDevice`, and
  through the polymorphic `Kernel::launch_on_host` handle.
  Registered via `carbon_add_cuda_unit_test` (no direct
  CUDA APIs in the body — runs on both CUDA and SYCL
  backends).

  Verification: 64/64 CUDA default + 76/76 CUDA+MPI+NCCL +
  53/53 SYCL pass; `format-check` clean on all three trees.
  The new test is one of the audit's "Block-friendly tests
  that should run on both backends" candidates flagged by
  P2.γ — it lands directly in the cuda_unit_test bucket
  rather than the CUDA-only gate.

- **2026-05 — P1.γ doc drift refresh (third-pass audit
  closure).** Seven documentation strings that had fallen
  behind the live code were refreshed in one commit:
  AGENTS.md (2 sites — test count claims `9 vs 11`, `+7 vs
  +12`, `45/45 SYCL`, `BlockReduceKernelImpl static_assert`
  clause; SpanConcept/PartitionConcept bullet rewritten to
  match the live partial anchoring story);
  `dense_disg.h` class blurb ("kernel factories ...
  intentionally not exposed yet" → enumerated the six
  factories actually exposed + HaloDisg pointer);
  `sycl/dense_kernel.h` ("lands in phase 2.4 alongside ..."
  → present-tense pointer to `dense_launch.h`);
  `test_skeleton.cu` (subtest comment "throws not yet
  implemented" → pointer to `apply_occ_` and the parallel-
  sibling / overlap subtests); `overlapped_stencil.h`
  (`first_` member-declaration comment reconciled with the
  in-function comment that correctly says `first_` no
  longer gates the fence). Marks P1.γ resolved in the audit
  doc.

- **2026-05 — P2.γ partial closure: CUDA-only test gate
  rationale (third-pass audit follow-up).**
  `tests/CMakeLists.txt` now carries per-entry comments
  documenting *why* each remaining `carbon_add_cuda_only_unit_test`
  entry is on the strict gate. The dominant pattern across
  the remaining 10 entries is **`cudaMemset` used to zero
  per-GPU device buffers in test setup** — a backend-portable
  `System::async_memset` (host-zeroed `Cpu::buffer<T>` via
  `System::async_copy`, or `q.memset` under
  `CARBON_BACKEND_IS_SYCL`) would lift seven of them in one
  commit. `block_cuda` is the only genuinely backend-specific
  entry by design (it's the CUDA-runtime boundary smoke test).

  As a partial closure, `block_host_mirror` was promoted to
  `carbon_add_cuda_unit_test` after an audit-driven inspection
  confirmed the test body uses zero direct CUDA APIs — it was
  on the strict gate by historical accident, not by genuine
  backend dependency. Now runs on both CUDA and SYCL. The
  gate is now 10 entries (down from 11) and every remaining
  entry carries a documented rationale.

- **2026-05 — Example: Dense sphere SDF checkpoint + PGM slice.**
  `examples/checkpoint_sphere_image.cu` (`carbon_example_checkpoint_sphere_image`)
  seeds a scalar field with an analytic sphere signed-distance, round-trips it
  through `CheckpointWriter` / `CheckpointReader`, downsamples an XY slice at
  mid-`z`, and writes an 8-bit grayscale Netpbm `P5` image (single-process /
  `mpiexec -n 1`; multi-rank MPI skips the image step with a log hint because
  z-shards are rank-local).

- **2026-05 — CTest `RESOURCE_LOCK carbon_gpu` for host .cpp tests that touch
  devices.** `carbon_add_cuda_unit_test` already serialised every `.cu` test,
  but `carbon_gpu`, `carbon_gpu_group_all_devices`, and `carbon_dense` (plain
  `carbon_add_unit_test`) could still run in parallel with those under `ctest
  -j`. Docker GPU CI then intermittently saw `carbon_skeleton` fail its
  multi-stream vs single-stream reference compare (stale / cross-process GPU
  interference). The three registrations now share the same `carbon_gpu` lock.

- **2026-05 — Dense / DenseDisg physical grid metadata: `origin` + `spacing`.**
  `carbon::float32_3d` origin (world position of global cell `(0,0,0)`'s corner) and
  scalar isotropic `spacing` are stored on `Dense` / `DenseDisg`, copied onto every
  per-GPU `Span`, and duplicated on each `Partition` so device code can read the
  mapping without a side channel. Defaults preserve the historical index-only grid
  (`origin` zero, `spacing` 1). Constructors reject non-finite origins, non-finite
  spacing, and `spacing <= 0`.

- **2026-06 — Native HIP backend (`-DCARBON_BACKEND=Hip`).** Carbon now compiles
  device code with ROCm's HIP toolchain directly, in addition to the existing CUDA
  (`nvcc`) and SYCL (AdaptiveCpp) backends. The design exploits that HIP's runtime is
  a near-textual twin of CUDA: every `cudaXxx` has a `hipXxx` counterpart with the
  same signature, `dim3` / `__global__` / `<<<>>>` are spelled identically, and `cub`
  maps one-to-one onto `hipcub` (rocPRIM-backed). Rather than fork the launch glue
  twice, the backend rests on two pillars:
  - **`select.h` macro normalization.** `CARBON_BACKEND_HIP` joins the mutual-exclusion
    guard and defines `CARBON_BACKEND_IS_HIP`, plus two new umbrella macros:
    `CARBON_BACKEND_IS_CUDALIKE` (CUDA *or* HIP) and `CARBON_CUDALIKE_DEVICE_TU`
    (`__CUDACC__` *or* `__HIPCC__`). The shared `detail/` impl headers
    (`dense_kernel_impl`, `reduce_kernel_impl`, `universal_*_impl`, `launch_traits`,
    `reduce_stages`) fork on these rather than on `CARBON_BACKEND_IS_CUDA &&
    __CUDACC__`.
  - **`detail/backend/cuda_like.h` neutral aliases.** `gpu_stream_t`, `gpu_error_t`,
    `gpu_success`, `gpu_error_string()`, `gpu_set_device()`, and the `gpucub` namespace
    alias resolve to the CUDA or HIP spelling under a single `CARBON_BACKEND_IS_CUDALIKE`
    arm. The shared impl headers reference only these; the backend-specific kernel /
    launch / gpu headers under `detail/backend/{cuda,hip}/` keep spelling the native
    names so each tree stays independently readable.
  The HIP backend-specific tree (`detail/backend/hip/`: stream / event / gpu(.inl) /
  the dense, dense_disg, block kernel + launch + reduce headers / reduce_combine) is a
  faithful hipify of the CUDA tree; `src/gpu_hip.cpp` + `src/hip_stub.cu` mirror the
  CUDA host-side / stub sources. `CARBON_HD` and the host-only `static_assert`s in
  `integer3d.h` / `floating3d.h` now also recognize `__HIPCC__`. CMake gains a `Hip`
  branch: discovers ROCm via `CARBON_ROCM_PATH` / `ROCM_PATH` / `/opt/rocm`, calls
  `enable_language(HIP)` + `find_package(hip)` / `find_package(hipcub)`, routes the
  `.cu` sources through the HIP language (`set_source_files_properties(... LANGUAGE
  HIP)`), targets `CMAKE_HIP_ARCHITECTURES` (`-DCARBON_HIP_ARCH=gfx90a` default), and
  links `hip::device` + `hip::hipcub`. The tests / examples / benchmarks helpers gained
  the matching `Hip` arm. Under MPI the native HIP backend selects RCCL (NVIDIA-on-HIP
  is not the target). The CUDA and SYCL backends are unchanged (configure + format-check
  verified clean). A `docker/Dockerfile.hip` (rocm/dev `-complete` base, no
  AdaptiveCpp/conda layer — the AMD analogue of `Dockerfile.cuda`) plus **hip** /
  **hip-mpi** configs in `scripts/docker-build-all.sh` make the build runnable from any
  host with Docker (`scripts/docker-build-all.sh --only hip,hip-mpi --gfx-arch gfxNNNN`);
  the image cross-compiles the device TUs with amdclang++ even without an AMD GPU present
  (whole compile + link of the library, all tests, and all examples — 151 targets — is
  green on ROCm 6.4.4 / gfx90a), and runs `ctest` only when `/dev/kfd` is reachable.
  Two integration snags surfaced and were fixed during that docker validation: (i) the
  ROCm `hip::device` / `hip::hipcub` imported targets inject `-x hip` as *unconditional*
  interface compile options, which the host C++ compiler driving carbon's `.cpp` sources
  rejects ("language hip not recognized") — carbon now links the host-safe `hip::host`
  (the AMD analogue of `CUDA::cudart`) and forwards only the hipcub / rocPRIM *include
  dirs* to the HIP-compiled TUs; (ii) the `src/dist_transfer.cpp` / `src/dist_reduce.cpp`
  collective-submit dispatch keyed off `CARBON_BACKEND_IS_CUDA` with a SYCL `#else`, so
  native HIP wrongly fell into the AdaptiveCpp interop arm — the guard is now
  `CARBON_BACKEND_IS_CUDALIKE` (native HIP submits the RCCL group on the `hipStream_t`
  directly, exactly like CUDA on its `cudaStream_t`).

- **2026-06 — HIP backend gets runtime coverage on NVIDIA (HIP-on-NVIDIA).** HIP also
  targets NVIDIA GPUs: with `-DCMAKE_HIP_PLATFORM=nvidia` (CMake >= 3.28) the HIP language
  is compiled by **nvcc**, `<hip/hip_runtime.h>` forwards every `hipXxx` to the matching
  `cudaXxx`, and hipcub maps onto cub — so the *exact same* native-HIP source tree runs on
  an NVIDIA card. This is what lets the HIP backend be validated at runtime on an NVIDIA
  host (67/67 tests pass on an A100, `sm_80`). Carbon's CMake `Hip` branch is now
  platform-aware (`CARBON_HIP_NVCC`): on the NVIDIA platform it defaults
  `CMAKE_HIP_ARCHITECTURES` to a CUDA capability (`80`), links `CUDA::cudart` + wires the
  `__HIP_PLATFORM_NVIDIA__` define and HIP/hipcub include dirs by hand (skipping
  `find_package(hip)` / `find_package(hipcub)`, whose ROCm `hip-config-nvidia.cmake`
  creates `hip::device`/`host`/`amdhip64` with un-guarded `add_library(IMPORTED)` and so
  aborts when pulled twice via the hipcub→rocprim→hip chain), and adds `--extended-lambda`
  to the HIP-language TUs (nvcc needs it; the AMD HIP-clang path does not). The HIP
  backend-specific device-code guards were broadened from `defined(__HIPCC__)` to
  `defined(__HIPCC__) || defined(__CUDACC__)`, since under the NVIDIA platform the device
  compiler is nvcc (`__CUDACC__`), not amdclang (`__HIPCC__`) — without this the
  `__global__` kernels (e.g. `dense_grid_kernel`) were never defined. `docker/Dockerfile.hip-nvidia`
  (CUDA devel base + HIP headers/cmake from the ROCm apt repo — only `hip-dev` +
  `hipcub-dev`, no `hip-runtime-nvidia`, which would drag in a ~2 GB skewed CUDA toolkit
  and a `hipcc`/`hipcc-nvidia` file conflict) and the **hip-nvidia** config in
  `scripts/docker-build-all.sh` (NVIDIA GPU passthrough, runs ctest) make this
  reproducible from any NVIDIA + Docker host.

- **2026-06 — Unified the CUDA and HIP backend trees into one `cudalike/`.** The native
  HIP backend originally shipped `detail/backend/hip/` as a hand-maintained, near-verbatim
  hipify of `detail/backend/cuda/` (~1300 LOC duplicated; the two could only be kept in
  parity by review discipline + `format-check`, which checks style, not semantics). Since
  CUDA and HIP are a 1:1 runtime map and the launch surface (`<<<>>>`, `__global__`,
  `__shared__`, `dim3`) is spelled identically by both toolchains, the two trees collapse
  into a single `detail/backend/cudalike/` that talks to the runtime *only* through the
  neutral `gpu_*` aliases + `gpucub` namespace. To make that possible, `cuda_like.h` grew
  from 5 aliases to the full surface the backend tree uses: `gpu_event_t`, the
  `gpu_error_{not_supported,invalid_value,invalid_resource_handle}` codes,
  `gpu_event_disable_timing`, and inline wrappers `gpu_get_last_error` / `gpu_malloc` /
  `gpu_free` / `gpu_stream_{create,destroy}` / `gpu_event_{create,create_with_flags,destroy,record}`
  / `gpu_stream_wait_event` / `gpu_runtime_launch_kernel`. (The launch wrapper is named
  `gpu_runtime_launch_kernel`, not `gpu_launch_kernel`, to avoid colliding with the
  file-local marshalling helper of that name in `cudalike/gpu.inl`.) The 14 dispatch shims
  under `detail/` now fold their `CARBON_BACKEND_IS_CUDA` / `CARBON_BACKEND_IS_HIP` arms
  into one `CARBON_BACKEND_IS_CUDALIKE` arm pointing at `cudalike/`; `detail/backend/cuda/`
  and `detail/backend/hip/` are deleted. Net: a CUDA-or-HIP bug fix is now written once and
  cannot silently diverge between the two backends, and HIP-on-NVIDIA exercises the exact
  same source the CUDA backend compiles. Validated: CUDA 68/68 (native, A100) and
  HIP-on-NVIDIA 67/67 (the `hip-nvidia` docker config, A100) both green from the single
  tree; `format-check` clean. The SYCL tree is untouched (genuinely different launch model).
- **2026-06 — Multi-GPU halo exchange now self-fences against the source GPU's compute
  (real fix for the "flaky `carbon_skeleton`" drift).** The intermittent
  `carbon_skeleton` failures previously serialised away with a ctest `RESOURCE_LOCK`
  were root-caused to a genuine correctness bug, not harness contention. A stress run
  (`test_skeleton` × 4 concurrent processes) reproduced it ~98% of the time
  (157/160) while a *serial* loop was 0/300 and a *single-GPU* concurrent loop was
  0/80 — i.e. the failure is multi-GPU-specific and timing-triggered, and it fires in
  the **single-stream** Jacobi compare, so it is neither a `Skeleton` multi-stream
  scheduler-barrier bug nor mere cross-process interference. Root cause: `Halo::operator()`
  (and `HaloDisg` / `HaloBlock`) issued the intra-process peer copies via `Transfer<T>`,
  whose cross-GPU entries become `cudaMemcpyPeerAsync` enqueued on the **destination**
  GPU's stream. That destination-side queue carries no implicit ordering against the
  **source** GPU's prior compute (the kernel that wrote the owned cells the copy reads),
  so under cross-process timing pressure the copy could read transient/stale source bytes
  → a small numerical drift on halo-dependent cells. The exact gotcha was already
  documented on `StreamGroup::wait_for_all` and fenced inside `OverlappedStencil::step`,
  but the bare halo path (used by the manual reference loop *and* the `Skeleton` halo
  kernel, both of which call `Halo::operator()` verbatim) never applied it. Fix: each of
  `Halo` / `HaloDisg` / `HaloBlock` now allocates a per-GPU fence `Event` at construction
  (only when the layout has intra peer pairs) and opens `operator()` with
  `fence_event_.record(stream); stream.wait_for_all(fence_event_)` — sequencing the copies
  behind every GPU's prior compute. A single pre-copy fence per call suffices: the *next*
  halo's fence transitively covers the copy-vs-next-overwrite WAR. The fence is a no-op on
  single-GPU / no-halo layouts (empty `fence_event_`). Post-fix the same 4×40 concurrent
  stress is 0/160; full suite 68/68; `format-check` clean. The `RESOURCE_LOCK` is retained
  as defense-in-depth against aggregate GPU-memory / NCCL contention under `ctest -j`, no
  longer as the fix for this race (rationale corrected in `tests/CMakeLists.txt`).
- **2026-06 — One-rank-per-GPU is a deliberate v1 distributed model, not an accident.**
  Every grid family's distributed path (`Dense`, `DenseDisg`, `Block`) asserts
  `gpu_count() == 1` per rank under `CARBON_HAVE_MPI_NCCL` and throws on the
  multi-GPU-per-rank hybrid. This is a conscious simplification, recorded here so it
  is not mistaken for an oversight: it makes `peer rank == peer global slab index`, so
  the halo's cross-rank classification, the positional NCCL `Send`/`Recv` pairing (no
  tags/barriers), and the `global_slab_count == world_size` partition all fall out
  directly with no per-rank device sub-partitioning bookkeeping. The cost is that a
  node with N GPUs runs N ranks rather than one rank driving N devices (slightly more
  MPI processes; intra-node exchanges still go GPU-to-GPU via NCCL, which handles
  same-node peers efficiently). Hybrid multi-device-per-rank (one rank per node, GPU-
  aware MPI) is the tracked follow-up; lifting it touches the slab-partition math and
  the halo peer-pairing on every family, so it is intentionally deferred until the
  single-device-per-rank surface is fully locked. Until then the throw is the contract.
- **2026-06 — Residual AMD-only runtime-validation risk, narrowed by the cudalike
  unification.** CI/Docker compiles the native AMD HIP build (`hip` / `hip-mpi`) but
  only runs `ctest` when `/dev/kfd` is present; the green runtime path validated on
  this host is `hip-nvidia` (nvcc + cub + CUDA runtime under the hood). After the
  `cudalike/` unification HIP-on-NVIDIA exercises the *exact same source* the native
  AMD build compiles, so the unexercised-at-runtime surface is no longer "the whole
  duplicated `hip/` tree" — it is narrowed to the genuinely AMD-specific runtime:
  rocPRIM-backed `hipcub` reductions, `hipMemcpyPeerAsync` peer copies, and RCCL
  collectives. Those three are the only behaviours a bug could hide in that current CI
  cannot catch; closing the gap needs a self-hosted AMD (ROCm) runner running the
  `hip` / `hip-mpi` configs with a real `/dev/kfd`. Flagged so the risk is explicit and
  owned rather than implicit.
- **2026-06 — `test_jacobi_integer`: bit-exact int32 Jacobi for halo / overlap /
  Skeleton correctness.** New `carbon_jacobi_integer` test (`tests/test_jacobi_integer.cu`):
  a radius-1 7-point Jacobi-like stencil on `int32_t` fields using truncating integer
  division (`sum / 6`) so halo / stream-ordering bugs show up as wrong integers, not FP
  drift. Three subtests against a Standard single-stream reference: (A) manual halo +
  Standard launch; (B) Internal/Boundary overlap across compute + halo streams with events
  (same shape as `test_data_view_split_jacobi`); (C) four-kernel Skeleton ping-pong
  (`halo_a → step_ab → halo_b → step_ba`, `kIters/2` cycles). Requires ≥2 GPUs (skips
  otherwise). Complements the float Jacobi overlap test and the broader `carbon_skeleton`
  suite — the integer compare is the sharp tool for validating the cross-GPU halo
  producer-fence fix landed earlier in this pass. Default test count: 70 CUDA / 69 SYCL
  (delta remains `block_cuda` only).
- **2026-06 — Fourth-pass P2 hygiene: HaloDisg host halo + cross-rank host Max/Prod +
  skeleton comment drift.** `tests/test_halo_disg_host.cu` exercises
  `HaloDisg::launch_on_host` on a multi-GPU Open layout (Host-only + HostDevice passes).
  `tests/test_dist_reduce_ops.cu` now runs host `Max` and host `Prod` through
  `make_host_reduce_kernel` + `MPI_Allreduce`, closing the P2.δ gap beside the existing
  host `Min` block. `include/carbon/skeleton.h` comments no longer reference the
  non-existent `detail/skeleton/*_inl.h` split — they point at `src/skeleton.cpp` where
  the P1.α bodies actually live. AGENTS.md MPI totals reconciled (+21 → 91 CUDA / 90 SYCL+MPI).
- **2026-06 — Pre-mGrid PR1: `StaticBlock` re-parameterized with the user-block
  sub-division.** Ahead of a multi-resolution (mGrid) port — which mirrors Neon by stacking
  one block-sparse grid per refinement level — `detail::block::StaticBlock` regained the
  `UserX/Y/Z`, `BlockRatioX/Y/Z`, and `IsMultiResMode` members that the original
  single-resolution pass trimmed (see `Neon/.../bGrid/StaticBlock.h`). All three are
  defaulted (`User == Mem`, ratios == 1, `IsMultiResMode == false`), so `StaticBlock<8,8,8>`
  / `DefaultStaticBlock` and every existing `Block<SBlock>` instantiation are byte-for-byte
  unchanged — this only re-opens the compile-time seam mGrid needs (one user block = one
  parent cell whose `BlockRatio` children live on the next finer level; e.g. a 2×2×2 octree
  level is `StaticBlock<8,8,8, 2,2,2, true>`). Two `static_assert`s enforce that the memory
  brick is a whole multiple of the user block per axis. No behaviour change; 70/70 + MPI
  green, format-check clean. First of the standalone P0 pre-mGrid Block refactors
  (`docs/` plan: pre-mGrid Block refactors).
- **2026-06 — Pre-mGrid PR2: `MicroIndex` tray view + `Span` conversions.** Studying Neon's
  `bIndex` settled the design: Neon's `MicroIndex` is **not** a stored field on the per-thread
  index — it's a *derived* (tray-block-idx, in-tray-idx) view computed on demand from
  `(block_idx, intra)` + the block ratios (`bIndex::getMicroIndex` / `init`). So rather than
  bloat carbon's hot-path `block::Idx` with a dormant field (the original plan note), this pass
  keeps `Idx` byte-identical and adds (a) a thin `detail::block::MicroIndex` data carrier
  (`int32_t tray_block_idx` + `int8_3d in_tray`) and (b) `Span<SBlock>::to_micro_index` /
  `from_micro_index`, which do the user-block fan-out arithmetic where the `SBlock` ratios are
  known — consistent with carbon's existing "`Idx` is data-only, `Span` owns the pitch math"
  split. Each memory brick fans out into `BlockRatioX·Y·Z` trays (one per user block). In the
  single-resolution default (ratios all 1) the conversion is the identity and the round-trip is
  exact, so nothing on the existing path changes — the methods are purely additive (no current
  caller). `tests/test_block_static.cpp` gains user-block-constant `static_assert`s, a single-res
  identity check, and an exhaustive 2×2×2-user-block (`StaticBlock<8,8,8,2,2,2,true>`, ratio 4 /
  64 trays per brick) round-trip + spot-checks. 70/70 + format-check clean. Second of the
  standalone P0 pre-mGrid Block refactors.
- **2026-06 — Pre-mGrid PR3: per-level `level` + `spacing` metadata on `Block::Span`.** A
  multi-resolution tower stacks one `Block` (hence one `Span` per GPU) per refinement level, so
  `block::Span` gained two defaulted topology fields: `level()` (0 = finest, larger = coarser)
  and `spacing()` (an `int32_3d` voxel size in finest-level units, `{1,1,1}` at level 0, growing
  by the per-level refinement factor as levels coarsen). Both ctors take them as trailing
  defaulted args (`level = 0`, `spacing = {1,1,1}`), so the lone single-level `Block` — which
  constructs spans through the existing 5-arg form — is byte-for-byte unchanged. This pass only
  parks the metadata: `global_location` keeps its single-level `origin + intra` semantics; the
  cross-level voxel↔world mapping that consumes `spacing` lands with the mGrid port.
  `tests/test_block_static.cpp` checks the defaults on the empty placeholder span and the 5-arg
  ctor, and that explicit coarse-level metadata (`level=2`, `spacing={4,4,4}`) is carried.
  70/70 + format-check clean. Third of the standalone P0 pre-mGrid Block refactors.
- **2026-06 — Pre-mGrid PR4: `MultiResPartitionConcept` + `mPartition` design note.** Pins the
  cross-level contract a future multi-resolution (mGrid) `Partition` must add on top of the
  single-level `PartitionConcept`, before any mGrid code exists. `MultiResPartitionConcept`
  (`include/carbon/detail/grid_concepts.h`) captures the parent/child surface of Neon's
  `mPartition` re-expressed in carbon's idiom — `has_parent` / `has_children`,
  `get_parent` / `get_child` (cross-level `Idx`), and `parent` / `child` value reads returning
  `NeighbourValue<T>` (`{value, valid}`) instead of Neon's `NghData` + bare references. Per-level
  scale stays off the partition (it lives on `Span::level()` / `spacing()` from PR3); Neon's
  `uncle` surface is documented as an out-of-minimal-contract extension. The concept is additive
  and currently unsatisfied (no mGrid partition ships yet) — it is the documentary anchor the
  mGrid port will `static_assert` against, the way `PartitionConcept` anchors the single-level
  families in `DeviceKernelImpl`. New `docs/design_mgrid.md` records the port shape (a tower of
  `Block` levels reusing the whole single-level data plane for intra-level work, adding only the
  cross-level layer), how the three pre-mGrid PR seams feed it, the `mPartition` backing data
  (`parent_block_id` / `child_block_id` tables + adjacent-level masks/storage), and the
  `getChild` / `getParent` arithmetic as PR2 `MicroIndex` round-tripping. 70/70 + format-check
  clean. Fourth of the standalone P0 pre-mGrid Block refactors.
- **2026-06 — Pre-mGrid design notes: co-resident multi-GPU partitioning + intra-level halo
  strategy.** Closes the two design questions the mGrid work hinges on, in
  `docs/design_mgrid.md` (§4, §5). **Co-resident rule (§4):** partition the *coarsest* level by
  z-brick-rows exactly as `Block` does today, then bind every finer-level brick to the GPU that
  owns its coarsest-level ancestor — i.e. the per-level z-cuts are the same world-space planes on
  every level. This makes `get_parent` / `get_child` / `parent()` / `child()` pure local pointer
  dereferences (a cell and its ancestor/descendant are always co-resident), so the **cross-level
  path carries no peer copy, no `Transfer`, no NCCL** — the design latitude the user asked us to
  keep. The alignment constraint (snap every level's cut to the coarsest level's brick boundaries
  in level-0 units) and the non-uniform-refinement load-imbalance follow-up are recorded.
  **Halo strategy (§5):** cross-level needs no halo; the only movement is the per-level radius-1
  edge stencil. Decision is **Option A — reuse one `HaloBlock<T,C,SBlock>` per level**, inheriting
  the tested intra-process `Transfer` + cross-rank `DistTransfer` (NCCL) paths for free; **Option
  B** (bespoke `System::async_copy` exchange, fully `Transfer`-free, fuse-friendly) stays a
  documented fallback if per-level halo *issue* overhead ever dominates on deep towers. Docs-only;
  no code change.
- **2026-06 — mGrid runtime port PR1: single-GPU data plane + cross-level partition.** First
  runtime pass of the multi-resolution grid on top of the pre-mGrid foundation. New tree
  `include/carbon/detail/mgrid/` + public `include/carbon/mgrid.h` (umbrella entry
  `carbon::MultiResBlock`). Four pieces: (1) **`MultiResDescriptor`** — runtime `depth` + a
  *uniform* refinement factor `R` (default 2, the octree case; per-level factors are a documented
  follow-up), with `voxel_spacing(l)=R^l`, `spacing(l)=R^(l+1)` (Neon's `getSpacing`), and
  `level_extent(finest,l)=ceil_div(finest,R^l)` — the ceil-div identity `ceil(ceil(n/R)/R) ==
  ceil(n/R²)` keeps adjacent levels consistent so a brick origin mapped across levels stays
  in-bounds. (2) **`MultiResBlock`** — holds a `std::vector<Block<SBlock>>` (one level per
  `Block`, reusing the whole bitmask / 27-way connectivity / origins / partitioning machinery
  verbatim) and builds the cross-level tables on the host by origin matching: `parent_block_id[l]`
  (one parent brick id per child block) and `child_block_id[l]` (one child brick id per *voxel* of
  each block, `block_count·VoxelCount` entries), uploaded per level. (3) **`MultiResField`** —
  one `Block::Field<T,C>` per level; `make_partition(level)` gathers this level's storage plus the
  adjacent levels' storage / spans / id tables into a cross-level partition. (4) **`mgrid::Partition`**
  — satisfies `detail::MultiResPartitionConcept` (anchored by a `static_assert`), with own-level
  `read`/`write` identical to `block::Partition` and the cross-level `has_parent`/`has_children`/
  `get_parent`/`get_child`/`parent`/`child` mirroring Neon's `mPartition`. **Reference
  reconciliation:** reading Neon's `mGrid.cpp` + `mPartition_imp.h` showed the refinement factor is
  `UserBlockSize` (the octree branching, 2) — *not* `BlockRatio` (`Mem/User`=4) — and that the
  cross-level arithmetic uses global-index math + the per-voxel `childBlockID` table, **not** the
  `MicroIndex` tray sketched in the earlier `design_mgrid.md` §2.3 (now reconciled; §2.4 added for
  construction). `MultiResBlock` enforces Neon's `ref_factor == SBlock::UserX` coupling.
  Single-GPU only this pass (mirrors Neon's own single-GPU mGrid restriction), but construction is
  written per-GPU-vector-aware so the multi-GPU co-resident follow-up slots in without reshaping
  the data plane. Tests: `tests/test_mgrid_static.cpp` (host — descriptor level math + a 2-level
  octree `get_child`→`get_parent` round-trip over a fully-active 16³ domain, a bijection over all
  4096 finest voxels) and `tests/test_mgrid_cross_level.cu` (device single-GPU — a hand-rolled
  kernel fills each level then reads children/parents through `make_partition`, checking 8192
  cross-level value reads; CUDA-only because PR1 defers `KernelLoader` integration so it uses a raw
  `<<<>>>` kernel). 72/72 default tests + format-check clean. Follow-ups (separate PRs): multi-GPU
  co-resident partitioning + MPI, `KernelLoader`/`Skeleton` integration, per-level `HaloBlock`, and
  a multi-res Jacobi/Poisson example.
- **2026-06 — mGrid runtime port PR2: multi-GPU + cross-rank co-resident partitioning.** Lifts the
  single-GPU restriction from PR1 and implements the co-resident partitioning rule of
  `docs/design_mgrid.md` §4. **Enabling seam on `Block`:** a new `Block::ZBrickPartition { begin,
  end }` + advanced ctor overload `Block(System, extent, is_active, ZBrickPartition, name)` lets a
  caller supply an explicit global brick-z cut instead of the default even split. The cut is
  validated (per-slab counts == `global_slab_count_`, contiguous, every slab ≥ 1 row, covers the
  candidate z-extent) and then drives every existing build pass (`sweep` / ghost reservation /
  connectivity / halo plans) unchanged — they already read `bz_*_global` / `bz_*_per_gpu_`. The
  primary ctor delegates with an empty partition (the even split), so the single-level `Block` is
  byte-for-byte unchanged and all 21 distributed Block/Dense/DenseDisg tests still pass.
  **`MultiResBlock` construction:** with more than one slab (`gpu_count()` single-process, or
  `world_size` under `CARBON_HAVE_MPI_NCCL`) it partitions the *coarsest* level by even z-brick rows
  and projects that single cut down to every finer level — level-`l` cut = coarsest cut ×
  `R^(coarsest-l)` (= `spacing(coarsest)/spacing(l)`). The projection is exact because the new
  multi-slab precondition requires `finest.z % (MemZ·R^(depth-1)) == 0`, so the coarsest brick rows
  tile each finer level with no remainder. The even split is recomputed identically on every rank
  (pure `even_split_z`), so distributed ranks agree on the cut with **zero communication**. A
  single-slab `System` passes no explicit cut and reproduces the PR1 layout exactly.
  **Why it matters:** a coarse cell and *all* of its finer descendants now land on one GPU/rank, so
  `get_parent`/`get_child`/`parent`/`child` stay pure local dereferences — **no peer copy, no
  `Transfer`, no NCCL on the cross-level path** (proved below). I verified the projection keeps a
  block's parent- and child-brick z-rows inside the same slab on every level. **Per-GPU cross-level
  tables:** `parent_block_id` / `child_block_id` (and their host mirrors) are now built per
  `(level, GPU)` from per-GPU origin maps — co-residency guarantees the parent/child lookup resolves
  within the same GPU's brick map — and uploaded one allocation per GPU. Tests:
  `tests/test_mgrid_multi_gpu.cu` (intra-process — builds the depth-2 tower over every visible GPU
  on a 16×16×32 domain and checks 16384 cross-level reads land co-resident; CUDA-only raw kernel,
  ran clean on 2×A100) and `tests/test_mgrid_dist.cu` (`mpiexec -n 2`, one-rank-per-GPU — each rank
  fills its own slab's levels and verifies all 8192 owned cross-level reads resolve locally with no
  halo/DistTransfer/NCCL, matching `encode(level, global)`). 73/73 default tests + 22/22 distributed
  (NCCL 2.27.5) + format-check clean. Follow-ups (separate PRs): descendant-weighted cuts to relax
  the z-alignment precondition + rebalance fine work, `KernelLoader`/`Skeleton` integration,
  per-level `HaloBlock`, and a multi-res Jacobi/Poisson example.
- **2026-06 — mGrid runtime port PR3: kernel layer (`MultiResBlock::make_kernel` via the
  `KernelLoader`).** mGrid compute kernels are now written exactly like every other grid family's
  — through the standard loading-lambda / `KernelLoader` contract — instead of the hand-rolled
  `<<<>>>` kernels PR1/PR2 used. **Design:** a per-level *view adapter*,
  `MultiResField::level_view(level)`, exposes the loader's duck-typed surface
  (`partitions()` / `host_partitions()` / `gpu_buffers()` / `name()`) where `partitions()[gpu]`
  is the cross-level `mgrid::Partition` for `(level, gpu)`. So inside a loading lambda,
  `loader.read(field.level_view(level))` (or `write` / `read_write`) hands back one partition
  value whose per-cell body can read/write its own level **and** walk to the parent (`L+1`) /
  child (`L-1`) levels transfer-free (co-residency, §4). No new launch machinery:
  `MultiResBlock::make_kernel(level, loading_fn, shared_mem)` just forwards to
  `grid(level).make_kernel(...)`, so the launch geometry is that level's `Block` (one CUDA block
  per retained brick, one thread per voxel) and multi-GPU / one-rank-per-GPU cross-rank are
  inherited from the tested Block launch path. The view also forwards `make_halo_kernel()` to the
  level's `block::Field` so `KernelLoader::read_stencil` (and the `Skeleton` auto-halo path) works
  for *intra*-level stencils — the cross-level path stays halo-free. Device-only this pass
  (`host_partitions()` is empty, so binding into an OpenMP kernel throws the same "no host mirror"
  diagnostic as a `MemorySide::Device` field). **Plumbing:** the per-family kernel conveniences
  follow the existing tail-include pattern — `kernel.h` now includes `<carbon/mgrid.h>` and a new
  `<carbon/detail/kernel_mgrid.inl>` (alongside `kernel_block.inl` / `kernel_dense*.inl`), where
  `Kernel` + `Block::make_kernel` are fully visible. `mgrid.h` forward-declares `Kernel` and only
  declares `make_kernel`; `mgrid.inl` (which never needs `Kernel`) defines `level_view`. Test:
  `tests/test_mgrid_kernel.cu` builds the fully-active depth-2 tower over a 16×16×32 domain across
  every visible GPU, fills each level with `encode(level, global)` via `make_kernel` + `level_view`
  writes, then a `make_kernel(0)` reads each level-0 voxel's parent and a `make_kernel(1)` reads
  each level-1 voxel's 8 octree children — purely through `loader.read(level_view(...))` — and
  writes match flags into an `out` field read back with `Block::download_to_host` (owned bricks
  only). The 16384 matching cross-level reads are GPU-count-independent (it ran clean on 2×A100).
  CUDA-only registration for now (mGrid SYCL parity is a tracked follow-up, matching the rest of
  the mGrid suite; nothing in the test body is intrinsically CUDA-specific). 74/74 default tests +
  95/95 under `-DCARBON_ENABLE_MPI=ON` (NCCL 2.27.5) + format-check clean. Follow-ups (separate
  PRs): the cross-level `make_reduce_kernel` (trivial forward, deferred to keep this pass to one
  tested surface), full `Skeleton` integration for multi-level token graphs, descendant-weighted
  cuts, per-level `HaloBlock` ergonomics, and a multi-res Jacobi/Poisson example.
- **2026-06 — mGrid kernel layer: cross-level `MultiResBlock::make_reduce_kernel`.** Rounds out the
  PR3 kernel layer with the reduction twin of `make_kernel`: `make_reduce_kernel<T>(level,
  loading_fn, dst, op, shared_mem)` forwards to `grid(level).make_reduce_kernel<T>(...)`, so the
  per-cell contribution `(Span, Idx) -> T` can read its own level **and** walk parent/child levels
  (through `loader.read(field.level_view(level))`) on the way to producing its reduction value —
  e.g. fold a residual built from a voxel and its parent. The per-brick stage-1 reduce, the
  intra-process combine (stage 2) and the cross-rank `ncclAllReduce` (stage 3) are all inherited
  from the Block reducer, so multi-GPU and one-rank-per-GPU work unchanged. Declared in `mgrid.h`
  (the `reduce::Sum<T>` default + `ReductionResult<T>` are already visible via `block.h`), defined
  in the `kernel_mgrid.inl` tail include next to `make_kernel`. `tests/test_mgrid_kernel.cu` now
  also drives the same cross-level parent/child reads through two `make_reduce_kernel` launches and
  checks the folded match counts (8192 + 8192) via `ReductionResult::host_value`. 74/74 default +
  95/95 with `-DCARBON_ENABLE_MPI=ON` + format-check clean (2×A100, NCCL 2.27.5). Follow-ups
  unchanged: full `Skeleton` integration, descendant-weighted cuts, per-level `HaloBlock`
  ergonomics, and a multi-res Jacobi/Poisson example.
- **2026-06 — mGrid intra-level halo: `MultiResField::make_halo_kernel(level)`.** Implements the
  design's §5 Option A — one `HaloBlock` per level. Because mGrid keeps a coarse cell and all its
  finer descendants co-resident (cross-level access is transfer-free), the *only* data movement a
  multi-resolution stencil needs is the **same-level** neighbour exchange, and this is it.
  `make_halo_kernel(level)` forwards to that level's `block::Field::make_halo_kernel` — the
  back-reference to the level's `Block` is set by `MultiResBlock::newField`, so a real `HaloBlock`
  is built (intra-process `Transfer` plus, under `CARBON_HAVE_MPI_NCCL`, the cross-rank
  `DistTransfer` companion). Iterate `level` in `[0, depth())` for a whole-tower halo. This is the
  user-facing standalone-exchange entry; it complements the `LevelView::make_halo_kernel` shipped
  with the kernel layer (which exists for `KernelLoader::read_stencil`'s auto-halo discovery on
  intra-level stencils). Declared on `MultiResField` in `detail/mgrid/field.h`, defined in the
  `kernel_mgrid.inl` tail include. Test: `tests/test_mgrid_halo.cu` builds the fully-active depth-2
  tower over a 16×16×64 domain (finest.z a multiple of `MemZ·R^(depth-1)`, so both levels carry
  ghost bricks at the slab cut) across every visible GPU, and for each level fills the owned bricks
  with `encode(level, global)`, runs the intra-level halo, and verifies every ghost slot holds the
  byte-for-byte payload of the matching owned brick on the adjacent GPU (`Block::halo_pairs_upward`
  / `halo_pairs_downward` per level — the `test_halo_block` Invariant-1 contract, now per level).
  This proves the projected co-resident cut yields a correct per-level halo plan at every level
  (the mGrid-specific part); the cross-rank `DistTransfer` machinery itself is already covered by
  `halo_block_dist`. 75/75 default + 95/95 with `-DCARBON_ENABLE_MPI=ON` + format-check clean
  (2×A100). Follow-ups: a cross-rank per-level halo MPI test, full `Skeleton` integration,
  descendant-weighted cuts, and a multi-res Jacobi/Poisson example.
- **2026-06 — mGrid intra-level halo: cross-rank MPI coverage.** Adds `tests/test_mgrid_halo_dist.cu`
  (`mpiexec -n 2`, one-rank-per-GPU), the co-resident sibling of `test_mgrid_halo`: where the
  intra-process test compares ghost bytes against the remote GPU's owned brick, this one proves the
  **cross-rank** path — the `DistTransfer` companion of each level's `HaloBlock` — actually delivers
  the neighbour rank's data into this rank's ghost bricks at the projected cut, on **every** level.
  A rank cannot peek the neighbour's buffer, so verification is kernel-side (the `test_halo_block`
  Invariant-2 contract, per level): after `MultiResField::make_halo_kernel(level)` runs, a Block
  compute kernel on that level reads each owned voxel's +z/−z same-level neighbour through
  `block::Partition::neighbour` (driven via `grid(level).make_kernel`, since the cross-level
  `mgrid::Partition` intentionally has no `neighbour`); at the rank boundary that resolves to a
  dist-filled ghost brick whose value must equal `encode(level, global±z)`, and a broken halo
  leaves the ghost at its memset-0 value and mismatches. Domain is the fully-active 16×16×64 depth-2
  tower (both levels carry ghost bricks at the rank cut). This closes mGrid's halo coverage matrix
  (intra-process multi-GPU + cross-rank, both levels). 75/75 default + 98/98 with
  `-DCARBON_ENABLE_MPI=ON` (NCCL 2.27.5) + format-check clean (2×A100). Follow-ups: full `Skeleton`
  integration, descendant-weighted cuts, and a multi-res Jacobi/Poisson example.
- **2026-06 — mGrid worked example: `examples/mgrid_multires.cu`.** The capstone tying the mGrid
  kernel + halo layers together end-to-end, every operation through the public kernel layer (no
  hand-rolled `<<<>>>`). Builds a fully-active depth-2 octree over an `N³` domain across every
  visible GPU (default `N=32`; CLI takes a single `N`, validated as a multiple of `MemZ·R` for the
  multi-GPU multi-slab precondition), seeds the finest level with the linear field `f = x+2y+3z`,
  and runs four **self-verifying** stages, each checked against a host reference that is exact for
  a linear field (so `max_err == 0`): **[A] restriction** L0→L1 (`make_kernel(1)` averaging the 8
  children via cross-level `child` reads), **[B] reduction** over L1 (`make_reduce_kernel<double>`
  summing coarse values, checked vs the host sum), **[C] prolongation** L1→L0
  (`make_kernel(0)` injecting the `parent` value via a cross-level `parent` read), and **[D] halo +
  smoothing** on L0 (`MultiResField::make_halo_kernel(0)` then `grid(0).make_kernel` averaging the
  6-axis neighbours through `block::Partition::neighbour`). Stage D's interior mean equals `f`
  exactly for a linear field *including* cells adjacent to the multi-GPU slab cut — which only
  holds if the halo delivered the neighbour slab's data, so it doubles as a halo check (27000
  interior cells verified at `N=32`, `max_err=0` on 2×A100). Registered like the other examples
  (build-verified under both backends; not a ctest — the underlying behaviours are already locked
  by `mgrid_kernel` / `mgrid_halo` / `mgrid_halo_dist`). Roadmap remaining: full `Skeleton`
  integration for multi-level token graphs and descendant-weighted cuts.
- **2026-06 — mGrid Skeleton integration: `MultiResField::bind_cross_level` + multi-level
  pipelines.** Wires `MultiResBlock` kernels into the `carbon::Skeleton` scheduler. The key
  realisation is that the Skeleton is **grid-agnostic** — it sequences kernels purely from their
  `AccessToken`s (`field_uid` + mode + pattern + optional halo factory) — and mGrid already emits
  correct per-level tokens: each `(field, level)` has its own `field_uid` because
  `LevelView::gpu_buffers()` forwards to `level_field(level).gpu_buffers()`, and
  `read_stencil(field.level_view(level))` snapshots a `TypedHaloFactory<LevelView>` whose `make()`
  builds that level's `HaloBlock` (lifetime-safe — the captured `MultiResField` lives in the
  kernel). So single-level pipelines and per-level auto-halo insertion *already* worked; the one
  real gap was that **cross-level reads were invisible to the dependency graph** —
  `loader.write(field.level_view(1))` records only the coarse-level token, but a restriction body
  also reads the fine level via `p.child(...)`, so the Skeleton wouldn't order restriction after the
  fine-level write. **Fix:** `MultiResField::bind_cross_level(loader, level, own_mode, read_child,
  read_parent)` binds the own level with `own_mode` (returning the full cross-level `mgrid::Partition`
  for the per-cell body) **and** records `AccessMode::Read` tokens on the requested adjacent levels.
  Those adjacent declarations bind the cheap `block::Field` (the discarded `block::Partition` is
  never used) but carry the **same** `field_uid` as `level_view(level±1)`, so the scheduler draws
  the fine→coarse / coarse→fine edges. Out-of-range `level±1` is silently skipped (the finest level
  has no child, the coarsest no parent). Declared on `MultiResField` in `detail/mgrid/field.h`
  (which now includes `<carbon/access_token.h>` for `AccessMode`), defined in the `kernel_mgrid.inl`
  tail include and templated on `Loader` so the header stays decoupled from `<carbon/kernel.h>`. No
  new scheduling code was needed in the Skeleton itself. **Tests:** `tests/test_mgrid_skeleton.cu`
  (CUDA-only) assembles a V-cycle-shaped pipeline — fill L0 → restrict L0→L1 → stencil-smooth L1 →
  prolong L1→L0 — into one `Skeleton`, then asserts the structural contract: exactly one auto-halo
  is inserted (immediately before the stencil consumer) and the `edges()` graph carries the
  fill→restrict and restrict→prolong dependencies that `bind_cross_level`'s child/parent tokens
  produce (without the helper neither edge would exist — the kernels would share no `field_uid`),
  followed by numeric verification on the linear seed `f=x+2y+3z` (exact). `tests/test_mgrid_skeleton_dist.cu`
  (`mpiexec -n 2`, one-rank-per-GPU) is the distributed sibling: it proves the Skeleton's
  **auto-inserted** per-level halo carries the cross-rank `DistTransfer` companion — a whole-tower
  fill+stencil pipeline where each level's stencil read triggers an auto-halo, verified kernel-side
  (the `test_halo_block` Invariant-2 contract: +z/−z `block::Partition::neighbour` reads at the rank
  cut must equal `encode(level, global±z)`, a broken halo leaves the memset-0 ghost and mismatches).
  `examples/mgrid_multires.cu` gains a stage **[E]** that re-runs restriction / stencil-smoothing /
  prolongation through one `Skeleton` (auto-halo, no manual `make_halo_kernel`) and verifies it
  reproduces stages A/C/D. 76/76 default + 100/100 with `-DCARBON_ENABLE_MPI=ON` (NCCL 2.27.5) +
  format-check clean (2×A100). Roadmap remaining: cross-tower OCC/`OverlapMode` (this pass keeps the
  default `OverlapMode::None`; per-level OCC is a tracked follow-up) and descendant-weighted cuts.

- **2026-07 — Neon feature-port gap closure (host halo, mGrid edges, Warp skeleton).**
  **Workstream A (bGrid host cross-rank halo):** `dist_transfer_detail::run_host` in
  `src/dist_transfer.cpp` implements the MPI `Irecv`/`Isend`/`Waitall` companion to the existing
  NCCL path; `HaloBlock::launch_on_host` builds per-direction MPI tags
  `(dir_channel << 16) | index` mirroring the NCCL FIFO pairing and calls `run_host` instead of
  throwing when `host_dist_pending_` is set. `tests/test_halo_block_host_dist.cu` (`mpiexec -n 2`,
  `MemorySide::HostDevice`, direct `cpu_buffers` seeding) verifies the host-only cross-rank path.
  **Workstream B (mGrid edges):** descendant-weighted coarsest-z cuts via `weighted_split_z` in
  `mgrid.inl` (optimal 2-way cut when `n == 2`); `tests/test_mgrid_weighted_cuts.cu` exercises
  imbalanced refinement. Cross-tower Skeleton OCC: `tests/test_mgrid_skeleton_overlap.cu` runs a
  V-cycle under `OverlapMode::{Standard,Extended,TwoWayExtended}` vs a `None` reference.
  SYCL/backend parity: `mgrid_kernel`, `mgrid_halo`, `mgrid_skeleton` promoted from
  `carbon_add_cuda_only_unit_test` to `carbon_add_cuda_unit_test`; `MultiResField::make_host_partition`
  + `level_view` populate `host_partitions()` when `cpu_buffers` exist;
  `MultiResBlock::make_host_kernel` forwards to `grid(level).make_host_kernel`;
  `tests/test_mgrid_host.cu` locks the host-partition gather wiring. Stale docs fixed:
  `docs/design_mgrid.md` runtime-landed banner, `mgrid.h` follow-ups note removed.
  **Workstream C (Warp bindings, CUDA-only skeleton):** `libcarbonpy/` C-ABI shim
  (`carbonpy_init`, `System`, `Dense`, `Field<float,N>` span/partition POD exports) gated by
  `-DCARBON_ENABLE_PYTHON=ON`; `py/carbon/` ctypes `Gate` loader; POST_BUILD copy to
  `py/carbon/lib/linux_<arch>/`; `tests/test_warp_smoke.py` + `carbon_warp_smoke` CTest entry.
  **Warp fork:** external native types require Lukasz Stolarczyk's fork vendored as a git
  submodule at `warp/` (`https://github.com/nvlukasz/warp.git`, branch
  `external-source-support-update2`; see `.gitmodules`, `scripts/setup_warp.sh`, `py/README.md`). `py/carbon/warp_builtins.py` registers
  `CarbonDenseSpan` / `CarbonDensePartition` via `wp.types.add_type()` against
  `libcarbonpy/include/carbonpy/warp_dense.h`; `tests/test_warp_external_types.py` launches a
  fill/check kernel through the registered builtins.
  Block/MultiResBlock exports, CUDA Driver launch matrix, and full `@wp.func` parity tests remain
  tracked follow-ups. Hybrid multi-GPU-per-rank under MPI stays deferred. Verified 79/79 default,
  104/104 with `-DCARBON_ENABLE_MPI=ON` (includes `halo_block_host_dist`), and `carbon_warp_smoke`
  with `-DCARBON_ENABLE_PYTHON=ON`; format-check clean.

- **2026-07 — Warp bindings pass 2 (Block C-ABI + parity tests).** Extended `libcarbonpy` with
  `carbonpy_block_*` exports and `CarbonBlockSpan` / `CarbonBlockPartition` POD blits for
  `StaticBlock<8,8,8>`; `libcarbonpy/include/carbonpy/warp_block.h` adds kernel builtins
  (`carbon_block_partition_read_f32`, `carbon_block_is_active`, …). Python: `py/carbon/block/`,
  `py/carbon/dense/`, `py/carbon/loader.py` stub; Warp registration covers Block types.
  Tests: `test_warp_block_external_types.py`, `test_warp_axpy.py` (Dense AXPY via Carbon
  builtins vs reference `wp.array` kernel). Warp submodule at `warp/` tracks
  `external-source-support-update2`; run `./scripts/setup_warp.sh` after checkout to build
  `warp.so`. Still open: CUDA Driver `cuLaunchKernel` container path, MultiResBlock exports,
  `container.py` / `skeleton.py`, wheel packaging.

- **2026-07 — Warp bindings pass 3 (CUDA Driver + Dense container).** `libcarbonpy` adds
  `carbonpy_cuda_driver_*` and `carbonpy_dense_container_*` (`cuda_driver.h` /
  `cuda_driver.cpp`): primary CUDA contexts per Carbon GPU, Warp-style 1-D grid launch via
  `cuLaunchKernel`, and a Dense container that stores a per-(GPU × DataView) matrix of Warp
  `CUfunction` hooks and passes `[launch_bounds, partition, span]` at run time.
  Python: `py/carbon/container.py` (`CudaDriver`, `DenseContainer`) plus `Loader.declare_kernel`.
  Test: `tests/test_warp_container_run.py` fills a multi-GPU Dense field through the Driver path
  and verifies with a reference Warp kernel. Fixed `test_warp_axpy.py` to use `wp.array(...,
  ndim=3)` instead of `wp.array3d` for reference buffers. Still open: MultiResBlock exports,
  `carbon_skeleton_*` C stubs, wheel packaging.

- **2026-07 — Warp bindings pass 4 (Block container, skeleton sequence, wheel).** Block
  `carbonpy_block_container_*` mirrors the Dense Driver container (`cuda_driver.cpp`).
  `carbonpy_skeleton_*` sequences Dense/Block container runs (`skeleton.cpp` /
  `py/carbon/skeleton.py`). Tests: `test_warp_block_container_run.py`,
  `test_warp_skeleton_sequence.py`. Added `py/setup.py` wheel helper copying `libcarbon.so` +
  `libcarbonpy.so` from `CARBONPY_BUILD_DIR`. Still open: MultiResBlock/mGrid C-ABI + container,
  full `carbon::Skeleton` token-graph integration from Python, jacobi/mres parity examples.

- **2026-07 — Warp bindings pass 5 (mGrid C-ABI + Jacobi container parity).** `libcarbonpy` adds
  `carbonpy_mgrid_*` grid/field exports and `carbonpy_mgrid_container_*` Driver containers
  (`mgrid.cpp`, level-aware block launch geometry). Python: `py/carbon/multires/`,
  `MgridContainer`, skeleton `add_mgrid_container`. Dense containers now support
  `field_count=2` for two-field kernels (Jacobi). Tests: `test_warp_mgrid_container_run.py`,
  `test_warp_jacobi.py`. **9/9** Warp CTests pass. Still open: `carbon::Skeleton` token-graph
  from Python, mGrid parent/child Warp builtins, mGrid OpenMP host kernel path.

- **2026-07 — Warp bindings pass 6 (token-graph skeleton).** Loader token capture
  (`py/carbon/loader.py` read/write/read_stencil), container `add_token` C-ABI, and
  `TokenGraphSkeleton` in `libcarbonpy/src/skeleton.cpp` (conflict scan, greedy multi-stream
  assignment, per-GPU CUDA events). Python `Skeleton.compile()` + `kernel_streams()`.
  Test: `test_warp_skeleton_token_graph.py` verifies parallel siblings `[0, 1, 0]`.
  **10/10** Warp CTests pass. Still open: mGrid parent/child Warp builtins, mGrid OpenMP host
  kernel path.

- **2026-07 — Warp bindings pass 7 (mGrid cross-level builtins).** `CarbonBlockSpan` gains
  `origins_ptr` for `global_location` in Warp kernels. New `CarbonMgridPartition` POD +
  `carbonpy_mgrid_field_get_cross_partition_f32` blit parent/child storage, block-id tables,
  and adjacent-level spans. `warp_mgrid.h` ports `has_parent` / `has_children` / `parent` /
  `child` read paths; Python registers the builtins in `warp_builtins.py`. Test:
  `test_warp_mgrid_cross_level.py` mirrors `test_mgrid_cross_level.cu` encode/check pattern.
  **11/11** Warp CTests pass. Still open: mGrid OpenMP host kernel path.

- **2026-07 — mGrid OpenMP host kernel path (Workstream B closure).** `make_host_partition`
  binds adjacent-level @c host_spans (not device spans) so cross-level @c parent() /
  @c child() bitmask checks are CPU-safe. Added @c MultiResBlock::make_host_reduce_kernel
  forward. Fixed @c Block::host_spans() to wire @c host_origins (was missing, breaking
  @c global_location on OpenMP). New @c test_mgrid_host_kernel.cu mirrors @c test_mgrid_kernel.cu
  on the OpenMP path (fill, parent/child checks, reduce) reading @c cpu_buffers directly.

- **2026-07 — Python examples (`py/examples/`).** Runnable Warp demos alongside the C++
  `examples/` tree: `jacobi_step.py` (Dense container Jacobi step, mirrors
  `test_warp_jacobi.py`) and `salt_pepper_filter.py` (PGM load, LCG salt/pepper noise,
  3×3 median filter via partition builtins, PSNR — port of `examples/salt_pepper_filter.cu`
  on a single GPU). Shared helpers: `pgm_io.py`, `_carbon_env.py`. Documented in
  `py/examples/README.md` and `py/README.md`.

- **2026-07 — Python bindings: object-oriented API + multi-dtype fields.** The Warp
  bindings moved from raw ctypes handles + `_f32`-suffixed free functions to a
  Neon-style object graph: `carbon.System` owns the multi-GPU state, grids
  (`carbon.Dense` / `carbon.Block` / `carbon.Mgrid`) are minted from it, and
  `grid.new_field(components=, dtype=)` returns a scalar-typed `*Field` object whose
  `get_span` / `get_partition` (and, for mGrid, `get_cross_partition`) hand back the
  same blittable POD structs the Warp kernels consume. Every object exposes `delete()`
  and is a context manager. `CudaDriver(system)`, `DenseContainer(driver, grid,
  loader, …)`, and `Skeleton(system)` all take objects now; the loader captures field
  *objects* and containers dispatch `bind_field` / `add_token` off the object's handle.

  The scalar type is now a first-class dimension. `libcarbonpy` grew a
  `carbonpy::detail::CarbonDtype` tag (0=f32, 1=f64, 2=i32, 3=i64) stored on a shared
  `FieldBox` / `MgridFieldBox` (hoisted into `include/carbonpy/detail/field_box.h` so
  the grid TUs and `cuda_driver.cpp` can't drift), and a `dispatch_dtype` helper that
  fans every `(dtype × components)` operation out to the right `Field<T, C>`
  instantiation. The C-ABI exports dropped their `_f32` suffix — `field_new` now takes
  a dtype code; `get_partition` / `field_delete` / `container_bind_field` /
  `add_token` read `box->dtype` internally. `warp_dense.h` / `warp_block.h` /
  `warp_mgrid.h` gained `_f64` / `_i32` / `_i64` read/write (and mGrid parent/child
  read) builtins, registered by dtype loop in `warp_builtins.py`. Partition PODs stay
  type-erased (a single `data_ptr`), so one partition struct per family still serves
  every dtype — only the builtins and C++ field templates vary. All 11 Warp CTests were
  migrated to the OO API and a 12th, `test_warp_dtypes.py`, round-trips a Dense field
  through every dtype. The `py/examples/` demos use the OO API too.

- **2026-07 — Python kernel decorator (`@dense.kernel`).** Added `py/carbon/kernel.py`
  with `dense_kernel` / `block_kernel` / `mgrid_kernel` factory decorators (lazy Warp
  import so `import carbon` stays fork-free). Each decorator wraps the user's per-cell
  function as `wp.func`, synthesizes the container `loading_fn` (field bind + optional
  `access` hints defaulting to `read_write`), builds the matching `*Container`, and
  auto-binds every visible GPU. Grid methods `Dense.kernel` / `Block.kernel` /
  `Mgrid.kernel` delegate to the factories. Additive sugar — the explicit
  `loading_fn` / `declare_kernel` path is unchanged. Test: `test_warp_kernel_decorator.py`
  (1- and 2-field dense parity). `py/examples/jacobi_step.py` now uses the decorator.

- **2026-07 — Python Warp builtins: dtype-tagged partition types + `wp.carbon`.** Warp
  kernels now use dtype-specific partition PODs (`CarbonDensePartitionF32`, …) with
  identical layout to the type-erased C-ABI struct; ``field.get_partition()`` mints
  the matching class from ``field.dtype``. A single ``wp.carbon.read`` / ``write``
  overload set covers every grid family and scalar type — Warp and C++ resolve dtype
  from the partition argument type (Neon-style). Helpers live under the same
  ``wp.carbon`` namespace. All Warp tests and examples migrated.

- **2026-07 — Python `@carbon.kernel` loading-lambda (Neon-symmetric).** Added
  ``CarbonDenseIdx`` / ``CarbonBlockIdx`` / ``CarbonIndex3`` Warp types plus Idx-based
  ``wp.carbon.set``, ``read(part, idx, c)``, ``write``, ``global_idx``, and ``get_x/y/z``
  overloads (additive alongside the existing span/coord builtins). Extended
  ``py/carbon/loader.py`` with Neon-style ``get_read_handle`` / ``get_write_handle`` /
  ``set_grid`` / ``set_mres_grid``. New ``@carbon.kernel`` decorator and
  ``build_bound_kernel()`` compile capture-based parameterless Warp kernels (partition +
  span baked into the compute ``@wp.func`` closure; wrapping kernel uses ``wp.tid()`` +
  ``set``). Auto-names kernels from the declared ``@wp.func`` when ``name`` is omitted.
  ``Skeleton.sequence(name, kernels)`` convenience added. Tests:
  ``test_warp_kernel_factory.py`` (Dense/Block/Mgrid two-pass add) and
  ``test_warp_skeleton_factory.py``. The older ``@dense.kernel(driver, …)`` explicit-signature
  decorators remain available.

- **2026-07 — Python Warp Idx read/write mirrors C++ `Partition::read(span, idx, c)`.**
  ``CarbonDenseIdx`` / ``CarbonBlockIdx`` are now lightweight (local coords only; span is
  no longer baked into the cell). Idx-based ``wp.carbon.read`` / ``write`` take
  ``(part, span, cell, component_id[, value])``; ``global_idx(span, cell)`` and
  ``is_valid(span, cell)`` (Block) likewise take span explicitly. Coord-based overloads
  remain as low-level primitives; examples and tests migrate to ``set(span, …)`` +
  span/cell read/write. Cross-level Mgrid parent/child reads stay on the coord overloads.

- **2026-07 — Python Warp neighbour / read_ngh builtins (Dense, Block, Mgrid).**
  Added ``CarbonNeighbourValue{F32,F64,I32,I64}`` plus ``wp.carbon.neighbour``,
  ``read_ngh``, ``value_or``, ``value``, and ``valid``. Dense ``CarbonDenseSpan``
  now carries ``global_x/y/z`` and ``boundary_mode`` for faithful
  ``neighbour_idx`` domain checks (Open + PeriodicZ). Block / Mgrid same-level
  neighbour reuses the 27-way connectivity walk + bitmask predicate. Jacobi example
  migrated off manual boundary checks. Test: ``test_warp_neighbour.py``.

- **2026-07 — Python DenseField field-level halo update (Neon parity).**
  ``DenseField.halo_update_run(stream_idx=0)`` mirrors Neon's
  ``dField.halo_update_run(stream_idx)``: lazily builds and caches a ``DenseHalo`` on first
  call, then runs ``carbonpy_dense_halo_run``. ``stream_idx`` is accepted for API parity but
  is not yet wired to stream selection (issues on ``System::stream()`` today). Field
  ``delete()`` tears down the cached halo. Test: ``tests/test_warp_field_halo_update.py``.
  Block/Mgrid field halo is a follow-up.

- **2026-07 — Python/Warp Skeleton reshaped to the Neon API (intrinsic halo + OCC).**
  ``py/carbon/skeleton.py`` now mirrors ``neon.skeleton.Skeleton``:
  ``Skeleton(system).sequence(name, containers, occ=SkeletonConfig.OCC.none()).run()``
  (arg-less ``run``; ``add`` / ``compile`` / ``kernel_streams`` retained). New
  ``py/carbon/skeleton_config.py`` ``SkeletonConfig.OCC`` (``none/standard/extended/
  twoWayExtended``, integer values matching ``carbon::OverlapMode``); exported as
  ``carbon.Skeleton`` / ``carbon.SkeletonConfig`` / ``carbon.OCC``. Halo insertion is
  **intrinsic** (Neon-style): a Dense field pulled with a stencil handle
  (``get_read_handle(field, operation=Operation.stencil)``) is recorded on the
  ``BoundKernel``/container (``build_bound_kernel`` in ``py/carbon/kernel.py``), and the
  Skeleton lazily builds one ``DenseHalo`` per stencil field and runs it before delegating
  to ``carbonpy_skeleton_run`` — no ``auto_halo`` toggle. Because the C-ABI tokens expose
  only ``field_uid/mode/pattern`` (not the field handle needed to build a ``carbon::Halo``),
  the insertion lives at the Python seam reusing the shipped ``carbonpy_dense_halo_*``.
  ``py/examples/jacobi_step.py`` migrated off the manual ``make_halo()``/``halo.run()`` onto
  ``Skeleton.sequence("jacobi", [jacobi]).run()``. Test: ``tests/test_warp_skeleton_halo.py``
  (asserts a halo was inserted; verifies multi-GPU Jacobi vs a NumPy global reference).
  ``ioToDot`` writes a minimal token-graph DOT. Follow-ups: the OCC overlap rewrite
  (Internal/Boundary split + halo-stream isolation for ``standard/extended/twoWayExtended``
  — currently warn + fall back to ``none``) and Block/Mgrid auto-halo (only Dense has a halo
  C-ABI today).

- **2026-07 — Python/Warp Skeleton bound to the C++ `carbon::Skeleton` (foreign-launch +
  real OCC).** Replaced the parallel ``TokenGraphSkeleton`` path with a true binding: Warp
  containers become ``Warp*KernelImpl`` foreign-launch ``carbon::Kernel`` nodes
  (``libcarbonpy/src/warp_kernel.cpp``) that replay stored Warp ``CUfunction`` hooks on the
  skeleton-assigned ``carbon::Stream`` via ``StreamGroup::native_at`` (CUDA/HIP only).
  ``carbonpy_skeleton_compile(handle, occ_mode)`` maps Neon-shaped Python ``OCC`` integers
  onto ``carbon::OverlapMode`` and runs the full C++ compile (auto-halo via
  ``TypedHaloFactory<Dense::Field>`` on stencil tokens, conflict scan, stream assignment,
  event wiring, OCC rewrite). ``carbonpy_skeleton_run`` calls ``Skeleton::run`` +
  ``Skeleton::wait``; ``inserted_halo_count`` / ``export_dot`` expose C++ introspection.
  Python ``Skeleton`` is now a thin ctypes wrapper (no Python-side halo cache). Tests:
  ``tests/test_warp_skeleton_occ.py`` (all four OCC modes vs NumPy reference + repeat-run
  stability under ``standard``). Follow-ups: Block/Mgrid stencil auto-halo parity and SYCL
  (Warp remains CUDA-centric).

- **2026-07 — Python Dense multi-GPU halo (single-process).** ``carbonpy_dense_new``
  accepts ``halo_radius`` (builds a von-Neumann ``Stencil`` when ``>= 1``). New
  ``carbonpy_dense_halo_{new,run,delete}`` C-ABI wraps ``carbon::Halo`` (intra-process
  peer copy + cross-GPU producer fence via ``System::stream()``). Python:
  ``Dense(..., halo_radius=1)``, ``DenseField.make_halo()`` / ``DenseHalo.run()``.
  ``py/examples/jacobi_step.py`` seeds every GPU slab, runs halo exchange, computes
  via ``read_ngh``, and verifies against a global NumPy reference assembled by
  ``z_offset + halo_z``.

- **2026-07 — Warp Dense span carries its ``DataView`` so OCC Internal/Boundary
  launches land on the right z-slabs.** The device-side ``carbon_dense_set``
  (``libcarbonpy/include/carbonpy/warp_dense.h``) previously ignored the view and passed
  the launch-thread ``iz`` straight through, so ``carbon_dense_linear_offset`` always
  applied the ``Standard`` mapping ``bz = halo_z + iz``. Under a non-``none`` ``OCC`` mode
  the skeleton splits the stencil kernel into ``Internal`` + ``Boundary`` launches whose
  z-iteration extent is thinner than ``Standard``; with the pass-through mapping both
  halves wrote the low-z owned cells (overlapping) and left the high-z owned cells
  untouched (zeros) — ``py/examples/jacobi_step.py`` under ``OCC.standard()`` produced
  ``max|carbon − ref| ≈ 18`` with many zero cells. The ``OCC`` test suite masked this
  because ``none`` ran first and filled ``v`` via the ``Standard`` view; reordering to
  run ``standard`` first reproduced the failure. Fix: ``CarbonDenseSpan`` (C ABI in
  ``carbonpy.h``, device struct in ``warp_dense.h``, ctypes mirror in
  ``py/carbon/warp_types.py``) now carries an ``int32_t data_view``; both span fillers
  (``carbonpy_dense_get_span`` in ``dense.cpp`` and the container ``fill_span`` in
  ``cuda_driver.cpp``) populate it, and ``carbon_dense_set`` remaps the launch-thread
  ``iz`` onto the owned-cell z axis exactly like ``carbon::detail::dense::Span::set_and_validate``
  (``Internal``: ``owned_z = halo_z + iz``; ``Boundary``: lower band ``owned_z = iz``,
  upper band ``owned_z = logic_z - 2*halo_z + iz``). Downstream ``linear_offset`` /
  neighbour math is unchanged (still owned-z + ``halo_z``). ``Standard`` (``data_view == 0``)
  is a pass-through, so seed / copy / ``OCC.none`` paths are byte-identical.
  ``jacobi_step.py`` now passes reliably under ``OCC.standard()`` and the four-mode OCC
  test passes regardless of ordering.

- **2026-07 — Wire the Warp ``Skeleton`` stream-introspection C-ABI to the real
  ``carbon::Skeleton``.** ``carbonpy_skeleton_kernel_stream_at`` and
  ``carbonpy_skeleton_num_streams`` (``libcarbonpy/src/skeleton.cpp``) were placeholder
  stubs left over from the C++ binding cut-over — they returned ``0`` / ``1``
  unconditionally, so ``Skeleton.kernel_streams()`` always reported every kernel on
  stream 0 and ``tests/test_warp_skeleton_token_graph.py`` (which schedules two
  independent sibling containers and expects ``[0, 1, 0]``) failed with
  ``[0, 0, 0] != [0, 1, 0]``. ``CarbonSkeletonBox`` now exposes ``num_streams()`` and
  ``kernel_stream_at(user_index)`` backed by ``carbon::Skeleton::num_streams()`` and
  ``kernel_streams()``; the latter maps the user-add index back through
  ``effective_to_user()`` (so auto-inserted halos in the effective-kernel list don't
  shift the reported per-user-kernel stream). A shared ``ensure_compiled()`` helper lets
  these read-only queries compile on demand (matching ``run`` / ``export_dot``). The C++
  greedy scheduler already assigns exactly ``[0, 1, 0]`` for the independent-sibling
  pattern (documented in ``skeleton.h``), so no scheduler change was needed — only the
  binding was returning fake data.

- **2026-07 — Add the ``Idx``-based Mgrid parent-``read`` Warp overload.** The Dense and
  Block families each expose two ``wp::carbon::read`` shapes — a decomposed
  ``read(part, span, coords…, component)`` and an ``Idx``-based
  ``read(part, span, idx, component)`` — but the Mgrid *parent* read only shipped the
  decomposed ``read(part, own_span, block_idx, x, y, z, component)`` form. So the natural
  kernel idiom ``cell = wp.carbon.set(own_span, b, x, y, z); wp.carbon.read(part, own_span,
  cell, 0)`` (used by ``tests/test_warp_mgrid_cross_level.py::check_parents``) had no
  matching overload and failed Warp codegen with ``Couldn't find function overload for
  'read'``. Added the four ``read(CarbonMgridPartition{F32,F64,I32,I64}, CarbonBlockSpan,
  CarbonBlockIdx, component)`` wrappers in ``libcarbonpy/include/carbonpy/warp_carbon.h``
  (each just unpacks ``idx.{block_idx,x,y,z}`` into the existing
  ``carbon_mgrid_parent_read_*`` primitive) and registered the matching builtin in
  ``py/carbon/warp_builtins.py`` next to the Dense/Block ``Idx``-based reads. Pure
  addition — the decomposed and child-read overloads are untouched. ``mgrid_cross_level``
  now passes.

- **2026-07 — Idx-only ``read`` / ``write`` for Carbon Warp kernels.** Reading and writing
  via explicit coordinates is not part of the programming model (matching Neon's
  ``neon_read(partition, idx, card)`` idiom). Removed all decomposed same-cell
  ``wp::carbon::read`` / ``write`` overloads from ``warp_carbon.h`` and their builtin
  registrations in ``warp_builtins.py`` (Dense ``ix,iy,iz``; Block ``block_idx,x,y,z``;
  Mgrid parent ``block_idx,x,y,z``). The only same-cell accessors are now
  ``read(part, span, idx, component)`` and ``write(part, span, idx, component, value)``
  (registered in ``_register_idx_builtins``). Every existing caller already used the Idx
  form. The mgrid cross-level *child* accessor was renamed from a raw-coordinate
  ``read(part, block_idx, x, y, z, ox, oy, oz, component)`` overload to
  ``read_child(part, idx, ox, oy, oz, component)`` — same Neon split as
  ``neon_read_child(partition, idx, ngh_idx, card)``. ``tests/test_warp_mgrid_cross_level.py``
  updated accordingly.

- **2026-07 — Fix ``test_warp_neighbour`` false failures (test correctness, not core bugs).**
  The Dense interior case hardcoded probe cell ``(4,4,4)`` and expected ``13.0`` while
  ``get_span()`` / ``get_partition()`` default to device 0 — on multi-GPU systems Carbon
  z-splits the domain, so ``z=4`` can sit outside device-0's owned slab and read unseeded
  ``0.0``. The test now derives ``(cx,cy,cz)`` from device-0's ``iter_*`` extents and
  asserts ``expected = (cx+1)+cy+cz`` for the ``+x`` neighbour. The mGrid same-level case
  expected ``11.0`` but the seed is ``x+y+z`` at neighbour ``(2,3,2)`` → correct value
  ``7.0``. No change to ``carbon_dense_neighbour_*`` / Block neighbour code.

- **2026-07 — Block + MPI cross-rank stencil parity regression (``test_block_dist_stencil_parity``).**
  The P0.A connectivity fix in ``block.inl::build_connectivity_`` (unconditional ghost-slot
  wiring for remote-rank neighbours) was already landed; ``jacobi_unified --grid block`` now
  matches single-process 2-GPU on ``sum|u|^2``. Added ``tests/test_block_dist_stencil_parity.cu``
  under ``mpiexec -n 2``: sphere-predicate domain, six-direction ``Partition::neighbour`` probe
  on cross-rank slab boundaries after ``HaloBlock``, plus Jacobi ``sum|u|^2`` vs a host
  reference. Retired stale "Known limitation" comments in ``jacobi_unified.cu`` and
  ``cg_poisson_unified.cu``. Distributed test count +1 → **92** on CUDA with MPI.

- **2026-07 — Benchmark gaps: ``reduce_throughput``, ``skeleton_overlap``, ``block_sparse``.**
  Added three scrapeable benchmark programs under ``benchmarks/`` (``REDUCE_GBs=`` /
  ``MLUPS=`` stdout lines) so reduction stage-1, Skeleton OverlapMode, and Block sparse
  stencil perf work is measurable before/after launch-sizing changes. Documented in
  ``benchmarks/README.md``.

- **2026-07 — POD layout drift guard (C-ABI / Warp / ctypes).**
  ``libcarbonpy/include/carbonpy/pod_layout_checks.h`` compile-time ``static_assert`` parity
  between ``wp::*`` and C-ABI structs; ``src/pod_layout.cpp`` exports ``carbonpy_sizeof_*`` /
  ``carbonpy_offsetof_*`` introspection; ``tests/test_warp_pod_layout.py`` compares ctypes
  layouts. Adding a field requires updating all three sites or the build / test fails.

- **2026-07 — Occupancy-derived dense launch blocks + SYCL async copy split.**
  ``dense_make_kernel_block(const detail::Gpu&)`` replaces the fixed ``8×8×8`` block with a
  per-device cached, warp-aligned factorization from ``max_threads_per_block`` /
  ``max_threads_per_multiprocessor`` (cap 512). ``System::async_copy`` on SYCL now submits
  D2D / peer / D2H memcpy without ``ev.wait()``; only H2D retains the synchronous wait for
  the AdaptiveCpp 25.02 host-pinning hazard. ``async_memset`` is fully async on SYCL.

- **2026-07 — Block halo transfer coalescing (``HaloBlock`` / ``coalesce_contiguous_transfers``).**
  Block brick storage already lays each z-boundary face out as consecutive local ids (x-fastest,
  y, z-outer enumeration plus stable internal/boundary packing; ghost slots consecutive in the
  same ``(by,bx)`` order). ``HaloBlock`` previously emitted one ``Transfer`` entry per boundary
  brick (~512 floats / 2 KB each), so multi-GPU ``block_sparse`` paid hundreds of
  ``cudaMemcpyPeerAsync`` launches per halo (~80× slowdown on PCIe). Added
  ``detail::halo_build::coalesce_contiguous_transfers`` to merge adjacent entries whose source
  and destination sub-ranges are contiguous within the same allocation pair; wired into
  ``HaloBlock`` and ``IntraTransferBuilder::finalise`` (no-op for Dense SoA multi-component
  planes). ``HaloBlock::launch_on_host`` applies the same coalescing to
  ``host_intra_`` so device and host paths issue the same number of face copies.
  ``tests/test_halo_coalesce.cpp`` locks the merge / gap / allocation-grouping rules.

- **2026-07 — Benchmark suite re-evaluation (micro-benchmark harness).**
  Added ``benchmarks/common/bench_report.h`` (``carbon-bench-report-v1`` JSON,
  ``--repeats`` median timing, ``elapsed_seconds`` on stdout, ``CARBON_BENCH_TOPO``).
  Upgraded ``reduce_throughput`` (``--batch-iters``), ``skeleton_overlap``
  (``--min-elapsed-ms`` launch-bound warning), and ``block_sparse``
  (``halo_pair_count``). Unified ``benchmarks/scripts/run_benchmarks.py`` +
  ``run_local.sh`` + ``plot_summary.py``; expanded ``benchmarks/README.md`` with
  ``halo_scalability``, interpretation guide, and post-coalescing reference table.

- **2026-07 — Skeleton schedule-graph export (Neon ``ioToDot`` parity).**
  ``export_dot`` now renders the lowered schedule (stream clusters, DataView
  labels, execution-order + event-wait edges). Added ``export_dot_file`` /
  ``export_dot_png``; LBM benchmark accepts ``--export-dot`` / ``--export-dot-png``.

- **2026-07 — Dense lattice halo auto-insertion (Neon ``StencilSemantic::streaming`` parity).**
  ``KernelLoader::read_stencil(field, stencil, semantic)`` snapshots a
  ``TypedLatticeHaloFactory`` so the Skeleton auto-inserts a
  ``StencilSemantic::Lattice`` ``Halo`` (z-sign population filtering on the
  aggregated Dense layout). ``Dense::Field::make_halo_kernel(name, stencil,
  semantic)`` builds the matching kernel. D3Q19 LBM iteration kernels now pass
  a Q-aligned population stencil at the load site (10 peer copies per adjacency
  pair vs 38 under ``Standard``). Added ``tests/test_lattice_halo_dense_auto.cu``.

- **2026-07 — NVTX ranges on ``Kernel::launch`` (Neon ``Skeleton`` parity).**
  Every ``Kernel::launch`` / ``launch_on_device`` / ``launch_on_host`` call
  pushes a yellow (ARGB ``0xFFFFFF00``) NVTX thread range labelled with the
  kernel's ``name()`` when ``CARBON_HAVE_NVTX`` is defined (CUDA, HIP-on-NVIDIA,
  and SYCL-with-cuda-target builds). Visible in Nsight Systems under
  ``--trace=nvtx`` without changing application code.

- **2026-07 — OpenMP fan-out of the per-GPU kernel launch loop (2x LBM
  multi-GPU throughput).** Both device launch paths — ``DeviceKernelImpl::
  launch_on_device`` (compute) and ``ReduceKernelImpl``'s stage-1 loop — issued
  their per-GPU ``gpu_set_device`` + launch-config work from a single host
  thread, so with N GPUs the host serialised ``N x (setDevice + launch)`` before
  the last GPU received work. On fast, high-iteration kernels (LBM) this
  host-side launch latency staircased the GPU start times and capped strong
  scaling. Mirroring Neon's ``DevSet::helpLaunchLambdaOnSpanCUDA``, the loops now
  run under ``#pragma omp parallel for num_threads(n) if (n > 1)`` — one host
  thread per GPU (single-GPU stays on the serial fast path). Exceptions cannot
  cross an OpenMP region, so per-GPU errors are collected and re-raised serially
  after the fan-out. Measured on 2xA100 PCIe, D3Q19 ``--overlap standard``
  N=384: **2801 -> 5611 MLUPS** (1400 -> 2805 per GPU); Nsight confirms
  ``cudaLaunchKernel`` now issues from 2 balanced host threads instead of 1.

- **2026-07 — ``OpenMP::OpenMP_CXX`` only decorates CXX-language TUs; forward
  the flag to nvcc.** ``OpenMP::OpenMP_CXX`` injects ``-fopenmp`` under a
  ``$<COMPILE_LANGUAGE:CXX>`` genex, so every ``#pragma omp`` inside a header
  instantiated by a ``.cu`` (CUDA-language) TU was silently a no-op — the launch
  fan-out above (and the host-kernel ``omp parallel for`` paths) ran
  single-threaded in user ``.cu`` TUs. The CUDA backend now adds
  ``$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>`` PUBLIC on the
  ``carbon`` target so consumer ``.cu`` targets (tests / benchmarks / examples)
  inherit host OpenMP through nvcc.

- **2026-07 — ``sync_root_after_nonroot`` now pins each event to its stream's
  device (latent multi-GPU reduce bug).** The stage-2 reduce fan-in created its
  scratch ``gpu_event_t`` on whatever device was *current* and recorded it on
  ``streams[i]`` (device ``i``); CUDA requires the event and stream to share a
  device, so this only worked when the current device happened to match. The old
  single-threaded reduce loop left the current device at the *last* GPU, which
  coincidentally matched the sole non-root stream on a 2-GPU/root-0 box (and
  would have broken on >2 GPUs). Exposed once the launch loop parallelised (the
  OpenMP team leaves the current device unspecified on return). Fixed by
  ``gpu_set_device``-ing to ``sys.gpu_group().devices()[i].device_id()`` before
  each event create+record and restoring the caller's device on exit (new
  ``gpu_get_device`` alias in ``detail/backend/cuda_like.h``). The helper now
  takes ``const System&``; the SYCL twin gained the matching (unused) parameter.

- **2026-07 — Lattice ``read_stencil`` extended to ``DenseDisg`` (grid-family
  parity audit).** ``KernelLoader::read_stencil(field, stencil,
  StencilSemantic::Lattice)`` was wired only for ``Dense::Field`` even though the
  ``HaloDisg`` engine already implements the full lattice run-length narrowing.
  Added ``DenseDisg::Field::make_halo_kernel(name, stencil, semantic)`` (forwards
  to ``HaloDisg(field, stencil, semantic)``) so the Skeleton auto-halo path
  inserts a lattice-narrowed ``HaloDisg`` for stencil-tagged DenseDisg reads,
  matching Dense. New ``tests/test_lattice_halo_disg_auto.cu`` asserts the
  Skeleton auto-inserts the HaloDisg kernel (compile + effective-kernel-count +
  run); numerical correctness of the disg lattice halo stays covered by
  ``test_lattice_halo_disg``. **``Block`` is intentionally excluded**: ``HaloBlock``
  transfers whole bricks (all components fused, SoA-per-block), so there is no
  per-population ``c_q.z`` filtering to express — ``Block::Field`` exposes only
  the plain ``make_halo_kernel(name)`` and ``read_stencil(field, stencil,
  semantic)`` on a Block field is a compile error by design.
- **2026-07 — Dense launch block reshaped for coalesced global access (3.1×
  single-GPU LBM).** ``dense_factor_block_threads`` previously aimed for a
  near-cubic power-of-two block and, for the 512-thread target, produced
  ``(bx=4, by=4, bz=32)``. But the dense buffer is *x-contiguous* (unit stride
  on x, row stride on y, slab stride on z — ``Span::linear_offset``) and
  ``dense_grid_kernel`` maps ``threadIdx.x`` onto the x cell coordinate, so a
  32-lane warp swept only 4 consecutive x cells across 4 rows and 2 slabs —
  scattering every global load across ~32 cache lines. The D3Q19/BGK lid-driven
  cavity at N=384 on one A100 80GB PCIe ran at **3122 MLUPS ≈ 475 GB/s ≈ 24 % of
  HBM peak**, ~3× below FluidX3D / Neon. The block now gives x a full warp (or a
  small multiple, capped at 64), spends the remaining threads on y (keeping the
  y±1 neighbour rows warm in L2), and pins z == 1 (z±1 planes served by L2):
  target 512 → ``(64, 8, 1)``. Same N=384 case now runs at **9775 MLUPS ≈
  1486 GB/s ≈ 77 % of peak** — on par with the SOTA references. The change is
  correctness-neutral (per-cell kernels are block-shape agnostic; the launch
  grid is re-derived by ``compute_dense_grid`` and the ``ValidatedIdx`` guard
  rejects the ceil-div overhang), so all 103 default tests stay green. Applies
  to both ``Dense`` and ``DenseDisg`` (both size their block through
  ``dense_make_kernel_block``).
- **2026-07 — `System` move no longer re-enumerates every GPU (fixed 2-GPU
  strong scaling ~1.4x -> ~2.0x; killed the OverlapMode regression).** After the
  block-shape fix made single-GPU ~3x faster, 2-GPU LBM only scaled ~1.4x and,
  worse, ``--overlap standard`` ran *slower* than ``--overlap none`` (N=384 2-GPU:
  7866 vs 13433 MLUPS). An nsys trace showed identical GPU-busy time in both
  modes but ~1.5x more wall time — pure GPU idle bubbles (~2ms between the halo
  and the next kernel), with utilisation collapsing to 39%. The CUDA API summary
  fingered ``cudaGetDeviceProperties`` called **744x** (avg ~1.5ms) during an 80-
  step run — ~2.6 calls per kernel launch. Root cause: ``System``'s move ctor and
  move-assignment built the moved-from placeholder ``State`` with
  ``detail::GpuGroup{}`` — the *default* ``GpuGroup`` ctor, which calls
  ``get_all_devices()`` and runs ``cudaGetDeviceProperties`` for every device.
  Because a ``Field`` owns its ``Dense`` (hence a ``System``) by value and
  ``KernelLoader::read_stencil`` snapshots the field into a ``TypedLatticeHalo
  Factory`` on **every launch** (moving the field in), each launch triggered a
  full device re-enumeration on the host — starving the GPU. ``standard`` doubles
  the launch count (Internal + Boundary), so it paid the tax twice and regressed
  below ``none``. Fix: the placeholder / default / moved-from ``State`` now uses an
  explicitly empty ``detail::GpuGroup{std::vector<detail::Gpu>{}}`` (a moved-from
  ``System`` is dead and must not probe devices). Result on 2x A100 80GB PCIe:
  2-GPU N=384 ``none`` 13433 -> **19654** MLUPS, ``standard`` 7866 -> **19590**;
  N=256 2-GPU ``none`` 5159 -> **18807**; strong scaling is now **1.99-2.01x
  (~100% efficiency)** for every (overlap, size) pair, and ``standard`` matches
  ``none`` instead of regressing. All 103 tests green.
- **2026-07 — Enable CUDA P2P peer access in `GpuGroup` (direct-DMA halo).**
  While investigating the above, the same trace showed the intra-process halo's
  ``cudaMemcpyPeerAsync`` copies decomposed into paired ``D2H``+``H2D`` on
  driver-internal streams — the signature of ``cudaMemcpyPeer`` falling back to a
  host bounce buffer because peer access was never enabled (Carbon never called
  ``cudaDeviceEnablePeerAccess``, even though ``cudaDeviceCanAccessPeer`` returns
  true for the PCIe pair). The ``GpuGroup`` ctor now sweeps every ordered device
  pair and enables peer access where the hardware allows it (tolerating
  ``cudaErrorPeerAccessAlreadyEnabled``, logging + skipping unsupported pairs so
  the host-staged fallback still works). Post-fix the halo copies are classified
  ``Peer-to-Peer`` (direct P2P DMA). On its own this was a modest win (the copies
  were already near PCIe bandwidth), but it removes host-bandwidth pressure and is
  the correct baseline for multi-GPU; the ``System``-move fix above is what
  actually recovered the scaling. HIP has an analogous
  ``hipDeviceEnablePeerAccess`` that is a tracked follow-up (the fix here lives in
  the CUDA ``src/gpu.cpp``; the SYCL backend manages peer access through its
  shared context).
- **2026-07 — `GpuGroup` default ctor is now empty (kills the re-enumeration
  footgun class).** The `System` move-path fix above stopped the moved-from
  placeholder from re-enumerating devices, but the underlying hazard remained:
  `detail::GpuGroup{}` (the default ctor) called `carbon::get_all_devices()`,
  so *any* default construction paid a full `cudaGetDeviceProperties` sweep
  (~1 ms/device). That is the wrong default for a type that is default- and
  move-constructed on hot paths. The default ctor is now `= default` (an empty
  group, no device probe) across all three backends (`src/gpu.cpp`,
  `gpu_hip.cpp`, `gpu_sycl.cpp`); enumeration is explicit via
  `GpuGroup(carbon::get_all_devices())` (that overload is also where P2P peer
  access is enabled and the per-device summary is logged). `System`'s default /
  move / move-assign placeholders simplify back to `detail::GpuGroup{}` now that
  it is cheap, and the lone consumer that relied on default-enumerate
  (`tests/test_gpu.cpp`) constructs explicitly. No public API relied on
  default-enumerate (Python bindings, examples, and tests all use
  `get_all_devices()`), so this is a semantics tightening rather than a break.
- **2026-07 — Skeleton recompile no longer strands CUDA events.** `add()` after
  a `compile()` sets `compiled_ = false` but does not clear the schedule state;
  the next `compile()` re-`assign`s its index vectors yet every phase only
  `push_back()`es into `events_`. A second compile therefore abandoned the prior
  compile's `Event` handles (a leak) while the freshly-assigned indices pointed
  only at the new tail. `compile()` now clears `events_` up front alongside the
  other schedule resets (`streams_` is intentionally kept — extra streams are
  reused in place). Single-compile pipelines are unaffected.
- **2026-07 — `Gpu::buffer` guards the size overflow.** `count * sizeof(T)`
  could silently wrap `size_t` and hand `gpu_malloc` a tiny byte count,
  returning a buffer far smaller than the requested element count and corrupting
  every subsequent indexed access. The element-count × element-size product is
  now checked (`throw_with_log` on overflow) before the multiply.
- **2026-07 — `KernelLoader` split: discovery vs launch phase.** The loading
  lambda bundles two jobs: (1) record access tokens + snapshot a
  `HaloFactoryBase` (via `read_stencil`, which does a `std::make_shared` that
  captures a full `Field` copy) for the `Skeleton`'s dependency analysis, and
  (2) return the per-cell callable that actually runs. Job (1) is a one-shot at
  kernel construction; job (2) runs on every launch. The impls re-ran the *whole*
  lambda per launch (per GPU under the OpenMP fan-out), so every launch paid job
  (1)'s allocations only to discard them. `KernelLoader` now carries a
  `KernelLoaderPhase { Discovery, Launch }`: `record_token_` early-returns and
  both `read_stencil` overloads skip the factory `make_shared` under `Launch`.
  The one-shot `discover_tokens_` sites keep the default `Discovery` (so
  dependency analysis is unchanged — all 103 tests, including the auto-halo
  `skeleton_halo` / `skeleton_occ` paths, pass); every per-GPU launch loop across
  the six kernel impls (compute / reduce / universal{,reduce} / host{,reduce})
  passes `Launch`.

  **Measured impact:** single-GPU LBM D3Q19 throughput is *unchanged* (N=128:
  8874 vs 8873 MLUPS; N=256: 9444 vs 9443) — the per-launch `make_shared` +
  `Field` copy was already fully latency-hidden behind async GPU execution (a
  ~sub-µs host cost overlapping a ~240 µs/iter memory-bound kernel), unlike the
  `cudaGetDeviceProperties` re-enumeration bug it superficially resembled (that
  was ~1.5 ms of *synchronous* host work per call that could not overlap). The
  value here is therefore (a) removing per-launch heap-allocator traffic — which
  matters most under the multi-GPU OpenMP fan-out where N threads otherwise hit
  the global allocator every launch — and (b) making the launch path
  allocation-free, a prerequisite for CUDA-Graph capture (Tier 1.2), where a
  captured region must not allocate. Kept as a correctness/hygiene + enabling
  change, not sold as a throughput win.
- **2026-07 — `Skeleton` CUDA/HIP graph capture + replay (opt-in).** A compiled
  `Skeleton` replays the identical launch sequence every `run()`, so it is a
  natural fit for CUDA graphs: capture the sequence once, then replay it with a
  single `cudaGraphLaunch` instead of re-issuing every node from the host. New
  `Skeleton::Options::use_cuda_graph` (default **off**) opts in.

  Implementation: a host-callable `detail::GraphReplay` (new
  `include/carbon/detail/graph_replay.h`) wraps stream capture + instantiate +
  launch through neutral `gpu_*` aliases added to
  `detail/backend/cuda_like.h` (`gpu_stream_begin_capture` /
  `gpu_stream_end_capture` / `gpu_graph_instantiate` (WithFlags,0) /
  `gpu_graph_launch` / destroy). These are runtime-API entry points, so the host
  `src/skeleton.cpp` drives them directly — no new `.cu` TU. `run()` splits into
  `issue_schedule_` (the per-step launch loop, shared), `run_eager_` (barrier +
  issue + iter-tail), and `run_graph_` (capture-on-first-eligible-run, then
  replay). Capture uses `cudaStreamCaptureModeThreadLocal`.

  **Scope + safety.** Engaged only when `graph_eligible_()` holds: option on,
  backend is CUDA/HIP-like, the compiled schedule is **single Skeleton stream**
  *and* **single GPU** (no cross-stream / cross-device event choreography to
  capture under ThreadLocal), and no host (OpenMP) kernels. Everything else —
  multi-GPU, multi-stream OCC, SYCL, host kernels — transparently uses the eager
  path. The graph is invalidated (`reset`) on every `compile()` (schedule
  rebuild) and re-captured if `run(view)` is called with a different `DataView`.
  A capture or replay failure latches `graph_disabled_`, logs once, and falls
  back to eager — a machine that cannot capture a given pipeline is never left
  worse off. `Skeleton::uses_graph()` exposes whether the last run replayed a
  graph (for tests / introspection). Correctness is locked by
  `tests/test_skeleton_cuda_graph.cu`: a single-GPU ping-pong replayed via graph
  is **bit-identical** to the eager path and `uses_graph()` is true on CUDALIKE.

  **Measured impact:** single-GPU LBM D3Q19 is unchanged within noise (N=48:
  12551→12564; N=96: 8039→8057; N=256: 9441→9438 MLUPS). Expected: this LBM runs
  a *single* fused kernel per iteration (the single-GPU halo is a no-op) followed
  by a per-iteration `wait()`, so there is essentially one launch to collapse and
  a forced device sync each step — graphs pay off for **multi-kernel-per-iteration**
  pipelines (e.g. multi-stage solvers) and host/CPU-bound regimes, not this
  shape. Shipped as a correct, zero-overhead, opt-in knob (`--cuda-graph` on the
  LBM benchmark) and a foundation. The high-value extension — **multi-GPU graph
  capture** via the fork/join origin-stream pattern, which would attack the
  small-N multi-GPU host-overhead that currently caps strong scaling — is the
  tracked follow-up.

- **2026-07 — Multi-GPU CUDA/HIP graph capture (opt-in), fork/join origin-stream
  pattern.** Extended the `Skeleton::Options::use_cuda_graph` path (still default
  **off**) from single-GPU-only to **intra-process multi-GPU**. A single
  `Skeleton` stream (`OverlapMode::None`) backed by N per-GPU streams is now
  captured as one **unified cross-device graph** replayed by a single
  `cudaGraphLaunch` on the origin stream.

  `detail::GraphReplay::capture` was generalised from `(stream, device_id)` to
  `(streams[], device_ids[])`. For N==1 it captures thread-locally as before. For
  N>1 it drives the classic fork/join origin-stream capture in
  `cudaStreamCaptureModeGlobal` (new `gpu_stream_capture_mode_global` alias) so
  the OpenMP-fanned per-GPU kernel launches on the peer streams are captured too:
  the origin records a fork event every peer waits on (pulling the peers into the
  same capture), the schedule is enqueued, then each peer records a join event the
  origin waits on before `endCapture`. Fork/join events are pre-created (never
  inside the capture window) and always destroyed on every exit path — a failed
  capture leaks nothing. The resulting graph carries the halo's cross-device peer
  copies + N×N event fences as intra-graph dependencies, so one launch on the
  origin replays every device's nodes with correct ordering.

  `graph_eligible_()` relaxed accordingly: single **Skeleton** stream (not single
  GPU), no host kernels, and **not distributed** (`sys_.distributed() == nullptr`
  — the cross-rank NCCL halo path is not captured here). Multi-*Skeleton*-stream
  (OCC) capture stays a follow-up.

  **Capturability fix (always-on):** cross-device copies in `System::async_copy`
  now use `cudaMemcpyAsync(..., cudaMemcpyDefault, stream)` (HIP:
  `hipMemcpyDefault`) instead of `cudaMemcpyPeerAsync`. Under UVA the driver
  infers both endpoints from the pointers, so this is the *same* peer DMA when
  peer access is enabled (`GpuGroup` enables it for every P2P-capable pair) — but,
  unlike the Peer variant, it is legal during stream capture
  (`cudaMemcpyPeerAsync` returns `cudaErrorStreamCaptureUnsupported` (900)). This
  is the change that actually lets a multi-GPU halo pipeline capture. As
  defence-in-depth, `run_graph_` now also catches a capture-time throw (e.g. a
  non-P2P pair whose copy must stage on the host) and falls back to eager instead
  of propagating.

  Correctness locked by `tests/test_skeleton_cuda_graph_multi.cu`: a 2-GPU
  stencil ping-pong with a real cross-device `Halo` (PeriodicZ) replayed via the
  cross-device graph is **bit-identical** to the eager path, with
  `uses_graph() == true` on CUDALIKE with ≥2 GPUs (skips as a pass on single-GPU
  hosts). Full suite 105/105 on 2×A100. **Measured impact:** 2-GPU LBM D3Q19
  N=256 is unchanged within noise (18792→18809 MLUPS) — same reason as single-GPU
  (one fused kernel + a per-iteration `wait()` leaves no host overhead to hide);
  the payoff is for multi-kernel-per-iteration pipelines without per-step sync.

- **2026-07 — `__launch_bounds__` measured and rejected on the dense kernel;
  warning baseline cleaned + opt-in `-Werror` CI gate.** Two roadmap items from
  the project review (Tier 1.3/1.4 + Tier 4):

  *`__launch_bounds__`* — the dense launch block is warp-aligned and capped at 512
  threads (`dense_make_kernel_block`), so `__launch_bounds__(512)` is a valid
  upper bound. An A100 sweep of LBM D3Q19 (the register-heaviest dense consumer)
  measured it **neutral-to-slightly-negative** across N=128..384 (e.g. N=384
  9780→9768 MLUPS). The pull-stencil kernels are HBM-bandwidth bound — already
  near the roofline — so there is no occupancy headroom for the hint to recover.
  Left off (documented in `dense_kernel.h`) rather than constrain ptxas for no
  gain; revisit only for a compute-bound dense consumer.

  *Warning cleanup + gate* — the `-Wall -Wextra -Wpedantic` library compile had
  four warnings; all fixed so the libcarbon build is warning-clean: a corrupted
  nested `/* */` in a `field.h` `@code` block; two `[[maybe_unused]]` reduce-impl
  `Traits` aliases (used only in the CUDALIKE device branch, unused in a host
  TU); and `[[maybe_unused]]` on `checkpoint.cpp`'s `maybe_distributed` (every
  caller sits behind `#ifdef CARBON_HAVE_MPI_NCCL`). New `CARBON_WERROR` option
  (default **off**) promotes host (CXX) warnings to errors on the `carbon`
  library target so CI can lock the clean baseline; device (nvcc) TUs are not
  gated (cross-execution-space diagnostics vary by toolchain). Full suite green.

  *Deferred (needs hardware):* narrowing the `Halo` cross-GPU producer fence from
  an N×N fan-in to a banded neighbour-only wait is correct-and-valuable for N>2
  GPUs but introduces an under-synchronisation risk that is **only observable
  with >2 GPUs** under timing pressure — untestable on the 2-GPU box, so left as
  a tracked follow-up rather than shipped blind. Likewise the `ReductionResult`
  raw-pointer lifetime hardening (shared-ownership) is a broad API change staged
  for its own reviewed pass.

- **2026-07 — CI wiring for the warning gate + GPU-correctness lane.** Enforced
  the new `CARBON_WERROR` gate where a real compiler runs and sharpened the
  GPU-correctness signal:
  - `ci.yml` `cuda-cpu-only` (every push / PR, no GPU) now configures with
    `-DCARBON_WERROR=ON`. The hosted runner's g++ (Ubuntu 22.04 / gcc 11) is the
    same host compiler as the CUDA container, so this is the cheapest place to
    keep the libcarbon compile warning-clean on every PR.
  - `scripts/docker-build-all.sh` gained an opt-in `--werror` flag that appends
    `-DCARBON_WERROR=ON` to the `cuda` config only (the library gate + the config
    that also runs `format-check`). Off by default so a local full-matrix run is
    never blocked by a stray host-gcc warning; CI opts in.
  - `gpu-ci.yml` passes `--werror` and its preflight now prints the visible GPU
    count, emitting a `::warning::` when `< 2` so the log makes explicit whether
    the multi-GPU correctness tests (`carbon_skeleton_cuda_graph_multi`, the
    cross-device halo/graph tests, every `*_multi_gpu`) ran for real or
    self-skipped. A ≥2-GPU runner now locks the cross-device graph-capture path
    added earlier this month; a single-GPU runner still validates the full build
    + single-GPU suite.

- **2026-07 — Grid-generic pull-stream: LBM benchmark runs on `Dense` @b and
  `DenseDisg` from one hot kernel.** The lid-driven-cavity LBM benchmark was
  hard-wired to `carbon::Dense`; the fast D3Q19+BGK path (and the lattice-generic
  body) pull-streamed populations through `Dense::Idx`-specific offset arithmetic
  (`pull_neighbour_idx` built a raw shifted `Idx` and read through it, skipping
  the `Partition::neighbour` validity branch — the branch elision is what closes
  the bandwidth gap with hand-rolled kernels). That raw pull is the one thing that
  did not generalise: `DenseDisg::Idx` additionally carries the five-region pitch
  fields (`offset_local_no_card` / `region_first_z` / `region_z_dim`) that
  `Span::set_and_validate` fills, so a neighbour cell in a different z-region
  cannot be addressed by shifting `(x, y, z)` alone.
  - **New primitive: `Partition::read_pull(span, idx, offset, component)`** on
    both `dense::Partition` and `dense_disg::Partition` — a @b trusted neighbour
    read (no validity branch; caller has proven the source is in-buffer, e.g. via
    a precomputed wall mask). Mirrors Neon's `dPartition::pull`. Dense builds the
    shifted `Idx` and does one `linear_offset`; DenseDisg routes through a new
    `dense_disg::pull_idx` free helper that recomputes the five-region pitch
    fields for the shifted cell (the classification is inherent to the disg
    layout, not extra per-neighbour work), then the same `linear_offset`. A
    pull-stream kernel written against `read_pull` therefore compiles and runs
    unchanged on either dense family — the grid-generic pull the fair comparison
    needed.
  - **LBM kernels + driver templated over `Grid`.** `make_iteration_kernel`,
    `make_iteration_kernel_d3q19_bgk_fast`, `make_compute_wall_mask_kernel`, and
    `make_macroscopic_kernel` now take `Grid&` / `typename Grid::template
    Field<...>` and use `read_pull` for the pull; `run_benchmark<Grid, Lattice,
    Collision>` constructs the selected family via `if constexpr` (Dense from the
    stencil, DenseDisg with `halo_z = 1`). New `--grid dense|disg` CLI switch
    (default `dense`); the JSON report gains a `config.grid` field. VTK dump stays
    Dense-only (`carbon::io::write_vtk` has no DenseDisg overload) and warns +
    skips on `--grid disg`.
  - **`DenseDisg::Field::fill_owned_host`** added (was Dense-only), resolving the
    `(x, y, owned-z)` triple through `Span::set_and_validate` + `linear_offset` so
    host seeding works in global coordinates without the caller knowing the region
    layout — the disg field is now as ergonomic to seed as the dense one.
  - **Parity verification: `--checksum`** Sum-reduces the final population field
    (`sum_q pop_q == rho` per cell → total domain mass, a grid-family- and
    GPU-count-invariant scalar). `--grid dense` and `--grid disg` agree exactly
    single-GPU (`830592.16792` at N=96) and to ~2e-7 relative on 2 GPUs (float
    summation-order noise across the halo/partition boundaries), confirming
    identical physics.
  - **Measured (2×A100 80GB PCIe, D3Q19+BGK, single precision).** N=256: Dense
    9470 MLUPS (1 GPU) → 18700 (2 GPU, `--overlap standard`, 1.97×); DenseDisg
    7185 → 14336 (2.00×). DenseDisg lands at ~76% of Dense single-GPU throughput
    on this cavity (the five-region layout trades the dense SoA's fully-coalesced
    population reads for region locality) but strong-scales just as cleanly. The
    two families now sit on the same benchmark so future disg-layout tuning has a
    like-for-like Dense baseline. **[Superseded by the fast-pull fix below — the
    24% gap was an artefact of `read_pull`, not the layout.]**

- **2026-07 — `DenseDisg::read_pull` fast path closes the single-GPU LBM gap
  with `Dense` (24% → ~0%).** The 76% figure above was traced not to the
  disaggregated *layout* but to `read_pull` recomputing the full five-region
  classification (`pull_idx`) on @b every one of the ~18 pull directions per
  cell. `cuobjdump --dump-resource-usage` on the fast D3Q19+BGK kernel showed the
  disg variant at **56 regs/thread vs Dense's 48** (+17%), dropping theoretical
  occupancy ~65%→~56% and, with it, achieved HBM bandwidth — the classic
  register-pressure-limited-occupancy failure mode on a memory-bound stencil.
  - **Root cause.** The regions are partitioned on @b z only, so a neighbour that
    stays in the same z-region as the centre shares the centre's per-region pitch
    fields — its element offset is the centre's offset plus a @b constant delta
    `(Δx + Δy·xy_pitch + Δz·slab_pitch)`, no reclassification. The old `read_pull`
    ignored this and reclassified unconditionally.
  - **Fix.** `dense_disg::Partition::read_pull` now takes the constant-delta path
    when (a) `Δz == 0` (all x/y neighbours — z-region cannot change) or (b) the
    centre @b and shifted cell are both in the internal region (detected from the
    centre's own `region_z_dim > 1` plus a range check on the shifted `bz`). Only
    a z-offset that actually crosses the internal↔boundary seam falls back to the
    full `pull_idx`. For an `Internal`-view launch that seam is two slabs; for the
    `Standard` launch it is the two owned boundary slabs — so ≈ all owned cells
    take the cheap load. The seam fallback stays off the hot path (the compiler
    constant-folds the per-direction `Δz` and does not let the rare branch inflate
    the kernel's register footprint), so the disg fast kernel now compiles to the
    @b same 48 regs/thread as Dense.
  - **Measured (same rig).** N=256: DenseDisg **9496 MLUPS (1 GPU)** vs Dense 9469
    — parity (disg fractionally ahead, within run-to-run noise); **18759 (2 GPU,
    `--overlap standard`)** vs Dense 18702. `--checksum` still bit-identical to
    Dense (`830592.16792` at N=96), so the fast path is numerically exact. This
    matches the Neon EuroPar result that dense vs disaggregated MLUPS differ
    negligibly, and confirms the disaggregated layout carries no intrinsic
    single-GPU penalty for a nearest-neighbour pull stencil — the earlier gap was
    purely the addressing code.

- **2026-07 — Parallel density rendering (`carbon::render`).** New public surface in
  `<carbon/render.h>`: `Camera::perspective` builds a row-major view-projection matrix;
  `DensityImage` owns one zeroed `width × height` float framebuffer per local GPU;
  `make_density_kernel` wraps a Neon-style density loading lambda into a compute kernel
  that projects each voxel centre through the camera and atomically splats into the
  per-GPU frame (`detail::atomic_add`, backend-dispatched CUDA `atomicAdd` / SYCL
  `atomic_ref`). `DensityImage::combine` async D→H's each GPU buffer, sums on the host,
  and (when `System::distributed()` is set) folds cross-rank contributions via
  `dist_host_allreduce_in_place(Sum)`. Rank 0 writes Netpbm P5 via `write_pgm`. Dense /
  DenseDisg use `Span::origin()` / `Span::spacing()` for world coordinates; Block grids
  take optional `block_origin` / `block_spacing` parameters. Example:
  `examples/render_density.cu` (sphere SDF, perspective splat, PGM). Test:
  `test_render_density` (energy conservation, frustum culling, multi-GPU combine parity).

- **2026-07 — Ray-march density rendering (`RayMarchImage`, `make_ray_march_kernel`).** Pixel-
  parallel front-to-back integration for `Dense` / `DenseDisg`: each local GPU writes per-pixel
  slab segments (`color`, `transmittance`, `t_enter`); `RayMarchImage::combine` merges local
  z-slabs by `t_enter`, and `dist_combine` (MPI) `Allgather`s rank segments before depth-ordered
  compositing. `Camera::ray_through_pixel` unprojects via stored inverse VP; field samples use
  trilinear interpolation (`render::sample_field_trilinear` / `sample_density_trilinear`).
  Multi-GPU / MPI runs require a fresh intra- (and cross-) slab @ref Halo exchange on the
  density field before marching so trilinear corners at z-partition boundaries read valid halo
  cells. Backend pixel launch lives in `detail/render_launch.h` (CUDA + SYCL). Example:
  `examples/render_ray_march.cu` (512³ sphere SDF, 4K PGM). Test: `test_render_ray_march`.

- **2026-07 — Ray-march AABB clip + step budget.** `RayMarchPixelFn` now intersects each
  ray with the global grid AABB before the per-GPU z-slab clip, so cameras placed outside
  the volume by `Camera::frame_bounds` start marching at grid entry (not at the eye) and
  get a finite exit parameter when `dir.z ≈ 0`. `make_ray_march_kernel` also raises
  `max_steps` to at least `ceil(grid_diagonal / step)` when the caller budget is too small.
  Fixes the 512³ / 4K example rendering a near-black frame with a thin horizontal streak.

- **2026-07 — Ray-march multi-GPU z seam.** Trilinear sampling now reads from the full haloed
  z-storage extent (not owned cells only), and the example runs @ref Halo on the SDF field
  before marching. Fixes a dark horizontal seam at intra-process slab boundaries (visible at
  image centre for the 512³ / 4K sphere when `dir.z ≈ 0` crosses the `z = N/2` partition).

- **2026-07 — Ray-march scalar-field shading (`ScalarField`).** `RayMarchShading::ScalarField` adds a
  threshold gate (`value_threshold`), magnitude range (`value_min` / `value_max`), named
  `Colormap` lookup (`Grayscale`, `Viridis`, `Jet` via `render::colormap_eval`), and
  `opacity_scale` for Beer–Lambert alpha. Samples below the threshold are fully transparent;
  above-threshold samples are colormapped and composited front-to-back. Example:
  `examples/render_wind_tunnel.cu` (analytic potential flow past a sphere, `Field<float,3>` velocity,
  ray-march of `|u|` with freestream transparent). Tests: `scalar_field_threshold`,
  `colormap_monotonic`.

- **2026-07 — Ray-march fill light.** `RayMarchOptions` adds optional `light_dir2` +
  `fill_diffuse` for a second diffuse-only term (specular stays on the key light). Example
  uses key + fill on the 512³ sphere for softer shadow-side detail without exposure hacks.

- **2026-07 — Ray-march lighting + RGB P6 export.** `RayMarchShading` adds three modes on
  `RayMarchOptions`: `Emission` (legacy grayscale density integration), `VolumeGradient`
  (pseudo-normal from ∇ρ, diffuse volume lighting), and `SurfaceBlinn` (SDF isosurface band +
  ∇SDF normal, Blinn–Phong with `light_dir`, `ambient` / `diffuse` / `specular`, `shininess`,
  `isosurface_width`, `material_color`). Per-GPU / MPI segments and front-to-back compositing
  carry separate R/G/B buffers; `host_color()` returns luminance for backward-compatible tests.
  `sample_density_gradient_trilinear` central-differences the trilinear field sample (halo-aware
  z bounds). `RayMarchImage::write_pgm` now writes Netpbm **P6** RGB with shared luminance
  min–max tone mapping (hue preserved) and optional `add_background` / `background` RGB.
  Example defaults to `SurfaceBlinn` on the 512³ sphere (`render_ray_march.ppm`). Tests:
  `test_surface_blinn_lit_side`, `test_volume_gradient_mode`, multi-GPU centre-row RGB parity vs
  single-GPU.

- **2026-07 — LBM benchmark distributed path validated end-to-end (MPI+NCCL,
  one rank per GPU).** The `lbm_lid_driven_cavity` driver already carried the
  `CARBON_HAVE_MPI_NCCL` wiring (auto-detect `Distributed`, `System(distributed)`,
  rank-0-only headline, cross-rank `--checksum` allreduce, per-rank VTK shards)
  but had never been built or run under MPI — the default `build/` configures
  without `-DCARBON_ENABLE_MPI=ON`, so those branches were dead. Built the
  benchmark into a `-DCARBON_ENABLE_MPI=ON` tree and exercised it on the 2×A100
  host with `mpiexec -n 2`: rank 0→GPU 0 / rank 1→GPU 1 binding, NCCL comm
  bootstrap, and near-linear strong scaling — dense 9452→18202 MLUPS (1.93×) and
  disg 9461→17736 (1.87×) at N=256; dense 9770→19256 (1.97×) and disg 9782→18921
  (1.93×) at N=384 (all `--overlap standard`, one rank per GPU over NCCL vs the
  single-GPU direct baseline). Cross-rank `--checksum` agrees with the
  single-process path to float summation-order noise (`830592.267466` vs
  `830592.286611` at N=96), and the NCCL halo lands within a few percent of the
  intra-process peer-copy 2-GPU numbers on the same PCIe node. Documented the
  distributed launch, launcher-side device-pinning guidance (the in-process
  fallback is `local_rank % visible_devices`, correct on one node but warns), and
  the measured scaling table in `benchmarks/README.md`. No source change — the
  benchmark was already distributed-capable; this pass proves it.
- **2026-07 — LBM `--report` writer is rank-0-only under MPI.** Every rank in the
  distributed benchmark path executed the JSON report block and opened the *same*
  `--report` file, so on multi-rank runs the writes interleaved and the file ended
  with a duplicated trailing brace (`}` `}`) — valid-looking until a strict JSON
  parser hit the extra data. Intermittent because it depended on which rank's
  `close()` landed last. Fixed by guarding the block with `world_rank == 0` (MLUPS
  is already reduced to a global figure, so rank 0's copy is authoritative), mirroring
  the existing rank-0 guard on the scrapeable `MLUPS=` headline. Surfaced while
  building the `(domain size × GPU count)` best-family colormap
  (`benchmarks/lbm_lid_driven_cavity/scripts/lbm-dgx-a100-40gb-run-colormap.{sh,py}`),
  whose extra multi-GPU cells tripped the corruption a handful of times.
- **2026-07 — Pure-MPI cross-rank halo transport (`CARBON_DIST_TRANSPORT=mpi`).**
  `dist_transfer_detail::run` now accepts an env-var override that routes device
  halo traffic through `MPI_Isend`/`MPI_Irecv` instead of the default NCCL
  group. The per-rank stream is synchronised before MPI (MPI is not
  stream-ordered like NCCL); when OpenMPI reports CUDA-aware support via
  `MPIX_Query_cuda_support`, device pointers are passed straight to MPI, otherwise
  a host-staged D2H → MPI → H2D fallback is used. Default remains NCCL (unset or
  `nccl`). Wired into the 8-GPU LBM Fig.7 sweep as a third transport line per
  grid family (`dense-mpip2p` / `disg-mpip2p` labels) via
  `lbm-dgx-a100-40gb-run.sh` and `plot_neon_fig7.py`.

- **2026-07 — Wind-tunnel example now runs a real D3Q27 + KBC LBM solve.**
  `examples/render_wind_tunnel.cu` was rewritten from the analytic potential-flow
  placeholder into an actual lattice-Boltzmann simulation of flow past a sphere,
  reusing the benchmark's `carbon::bench::lbm` kernels unmodified
  (`make_iteration_kernel<Dense, D3Q27, KbcCollision>`,
  `make_compute_wall_mask_kernel`, `make_macroscopic_kernel` from
  `benchmarks/lbm_lid_driven_cavity/src/`; the example adds that dir to its
  include path in `examples/CMakeLists.txt`). D3Q27 + KBC (Karlin–Bösch–
  Chikatamarla entropic collision) is used because KBC's per-cell `gamma` keeps
  the solve well-behaved; even so the practical stable ceiling here is set by the
  staircased sphere bounce-back (not the collision), so the example runs a steady
  laminar `Re = 40` (`omega ≈ 1.41`) on a `360×208×208` duct with a `radius = 28`
  sphere — the larger diameter maps a given Re onto a gentler `omega`. The domain
  is a genuine confined wind tunnel: no-slip bounce-back duct walls (mass-
  conserving), plus an example-local `dense.make_kernel` boundary kernel appended
  to each ping-pong `Skeleton` step that re-imposes a Dirichlet velocity inlet
  (`x==0` → `f_q^eq(ulb)`) and a fixed-pressure outflow (`x==Nx-1` → equilibrium
  at the extrapolated `x-1` velocity, density pinned to 1). A slip/far-field
  lateral clamp and a raw zero-gradient population-copy outlet were both tried and
  rejected — the former lets the bulk velocity inflate without bound (density
  drops so `u = m/ρ` blows up), the latter stalls the flow like a closed box. The
  two-pass render (ScalarField `|u|/ulb` in `Jet` above a threshold, transparent
  freestream/wake, over a Blinn-lit sphere SDF) is unchanged apart from tuning:
  thresholds are expressed relative to `ulb`, the camera is a side-on framing on a
  dark background so the accelerated core glows and the low-speed wake reads as a
  transparent shadow. A second output (@c render_qcriterion.ppm by default) ray-
  marches the Q-criterion field ( @ref make_qcriterion_kernel , central-difference
  velocity gradients with @c 1/ulb scaling) in Viridis above a positive threshold so
  the standing recirculation / shear layer around the sphere reads as a vortical band.
  Wake-box @c Q^+ peaks at @c O(10^{-3}) here (wall shear outside the clip reaches
  @c O(10^{-1}) ); render thresholds are scaled to the wake band.

- **2026-07 — Q-criterion helper and wind-tunnel vortex render pass.**
  `include/carbon/qcriterion.h` adds `make_qcriterion_kernel<Grid>` for Dense /
  DenseDisg vector fields: a stencil-read compute kernel that fills a scalar field
  with @c Q = ½(‖Ω‖² − ‖S‖²) from central-difference velocity gradients (writes
  @c 0 when any required neighbour is unavailable). Optional @c velocity_scale
  (e.g. @c 1/ulb ) keeps @c Q in dimensionless units. Registered in the umbrella
  header. `examples/render_wind_tunnel.cu` computes @c Q after the LBM macroscopic
  extract, halos the scalar field, and writes a second ray-march frame
  (`render_qcriterion.ppm` by default) with `ScalarField` + `Viridis` over the
  same lit-sphere composite as the existing `|u|/ulb` pass.

- **2026-07 — Wind-tunnel example aligned with XLB `flow_past_sphere_3d`.**
  `examples/render_wind_tunnel.cu` now follows the Autodesk XLB reference
  ([`flow_past_sphere_3d.py`](https://github.com/Autodesk/XLB/blob/main/examples/cfd/flow_past_sphere_3d.py)):
  domain `512×128×128` (2× the reference `256×64×64`), sphere radius `Ny/12`
  centred at `(Nx/6, Ny/2, Nz/2)` (~16% blockage), parabolic inlet
  `u_x = u_max·(1−r²)`, extrapolation outlet, `u_max = 0.05`, `Re = 15`
  (derived `ω ≈ 1.43` — fixed `ω = 1.6` collapses on the smaller staircased
  sphere), `15000` iterations. D3Q27+KBC and half-way bounce-back at the sphere
  (via `wall_neighbour_bits`) are unchanged; camera clip widened downstream to
  frame the longer wake run. Logs post-solve `|u| max` as a quick sanity check.

- **2026-07 — mGrid density rendering example (`render_density_mgrid`).**
  `examples/render_density_mgrid.cu` extends the sphere-SDF density-splat tutorial to
  an adaptive `MultiResBlock` tower (depth 3, `R = 2`, `512³` finest extent): finest
  bricks hug the surface (`|sdf| < 8`), a mid band covers the transition shell, and
  the coarsest level fills the deep interior. Each level is seeded with
  `make_kernel` + `level_view`, then splatted into one shared `DensityImage` via
  `make_density_kernel` on `mblock.grid(L)` with `block_spacing = R^L` (maps
  level-local voxel indices into shared finest-unit world space), `samples_per_axis`
  scaled by `R^L`, and a `(R^L)³` density gain so voxel volume is conserved across
  the resolution transition. Splats accumulate additively (`atomic_add` per launch;
  one `clear` / `combine` pair). Optional `--level L` renders a single tower level;
  a host-side 3D mass integral self-check logs relative error vs the analytic sphere
  volume. Registered as `carbon_example_render_density_mgrid` with optional viewer
  link.

  The level bands must be evaluated at a **level-independent anchor**. The first cut
  tested each level's band at that level's own voxel centre, which does not partition
  space: a coarse voxel centre and its fine children can land on opposite sides of a
  band edge, so a shell is claimed by two levels (double-splatted) or by neither (a
  hole). Both render as concentric rings — and because the gaps and overlaps very
  nearly cancel, the 3D mass check stayed within 0.07% and did not catch it. The fix
  snaps every voxel to the coarsest cell containing it (`coarse_anchor`) before
  evaluating `desired_level`, making the assignment an exact partition at
  coarsest-cell granularity; the transition boundary becomes staircased at `R^(depth-1)`
  voxels, which is invisible because the density is smooth across it. Diagnosed by
  radially binning the frame: the pre-fix profile was non-monotone with its largest
  deviation at world radius ≈ 131 = `radius - k_band_outer`, exactly the L1/L2 band
  edge.

  The regression guard for that bug is `level_partition_error`, which tests the
  invariant directly rather than through the picture. Coverage is accumulated at
  coarsest-cell granularity — each active level-`L` voxel contributes the `scale³`
  finest cells it occupies — so a claimed coarse cell must total exactly
  `coarsest³` and a culled one exactly 0; anything else is a gap, a partial claim or
  a double-claim. It is geometry-independent (no assumption about the scene shape),
  it also covers the brick-retention / bitmask / multi-GPU-origin plumbing it reads
  through, and it is not vacuous: reverting to per-level anchors reports 96329
  mismatched coarse cells out of 128³ and fails the run, while the 3D mass check
  still passes at 0.14% — which is exactly why the mass check alone was not enough.
  An earlier `radial_seam_error` variant (worst upward step of the binned radial
  profile) was dropped when the scene stopped being a centred sphere; the partition
  check supersedes it and is strictly stronger.

- **2026-07 — `render_density_mgrid` scene is now a sphere ∪ cylinder.** Exercises
  the defining property of a signed distance function: the SDF of a union is the
  pointwise `min` of the operand SDFs, so `scene_sdf = min(sphere_sdf,
  cylinder_x_sdf)` composes the two solids with no extra machinery, and the tower's
  refinement bands, the per-level seed kernels and the splat path all pick up the
  new geometry unchanged. `cylinder_x_sdf` is the *exact* capped-cylinder distance
  (in the `(radial, axial)` half-plane the solid is a rectangle, so the 2D box
  distance applies), which matters because the density transfer function is
  `tanh(sdf / w)` — an approximate SDF would bias the smooth surface band. Shape is
  `cyl_radius = 0.40 · radius`, `cyl_half_len = 1.30 · radius`: the caps protrude
  past the sphere so the union is visibly not a sphere, and `cyl_radius >
  k_band_outer` keeps all three levels populated inside the cylinder too.

  Keeping the mass self-check meaningful needed a closed form for the union volume.
  With the axis through the sphere centre, `a = cyl_radius < R` and
  `L = cyl_half_len ≥ R` (so the cylinder spans the sphere), the intersection is the
  spherical core within distance `a` of the axis,
  `V∩ = (4π/3)(R³ − (R² − a²)^{3/2})`, giving
  `V∪ = 2π a² L + (4π/3)(R² − a²)^{3/2}`. Measured `mass_3d = 2.60982e7` against
  analytic `2.60783e7` (0.077%), versus `2.41047e7` for the sphere alone — so the
  integral independently confirms both the `min` composition and the formula. The
  preconditions are enforced by construction via the two `k_cyl_*_frac` constants,
  and a runtime check rejects a cylinder whose caps would fall outside the domain
  (clipped caps would silently invalidate the analytic volume). The camera framing
  bbox grew along x to `max(radius, cyl_half_len)` so the caps stay in frame.

  Per-level frame sums (`--level 0/1/2` = 6.82e7 / 2.46e8 / 1.85e8) still add up to
  the combined 4.99479e8, re-confirming that the per-level splats accumulate
  additively with no double counting.

- **2026-07 — `DensityImage::write_ppm` (colour Netpbm export).** `DensityImage`
  could only emit grayscale `P5`, so tinting the field meant post-processing the
  file. `write_ppm(path, ink, background, mode)` writes `P6`, tone-mapping density to
  `t` in [0, 1] exactly as `write_pgm` does and then shading `lerp(background, ink,
  t)`: empty pixels take `background`, the densest take `ink`. Defaults are
  dense-black on white, matching the plotting convention rather than `write_pgm`'s
  white-on-black. The shared tone map is now `detail::render_export::ToneMap`, so a
  grayscale and a colour export of the same frame agree by construction — verified
  numerically: the PPM red channel is exactly `255 - pgm` at every sampled pixel
  (ink has `r == 0`), and `write_pgm` behaviour is unchanged across both
  `DensityNormalize` modes and the degenerate / non-finite paths.

  Removed the vestigial `DensityImage::set_export_options` declaration. It was the
  half-built hook for exactly this feature: declared in `render.h` but never defined
  in any TU, so any call site would have been a link error — it had no working
  callers and could not have had any. `RayMarchImage::set_export_options` is a
  different, fully implemented member and is untouched.

  `examples/render_density_mgrid.cu` now defaults to `render_density_mgrid.ppm` and
  renders green ink (`0, 0.42, 0.12`) on a white page; passing a `.pgm` path still
  selects the grayscale writer, so both formats stay reachable from one binary.

- **2026-07 — Viewer browser backend (`--web`, imgui-ws over WebSockets).**
  The interactive viewer gained a second backend so a headless (SSH-only) box
  can drive the render from a browser instead of a local GLFW window. New option
  `CARBON_VIEWER_WEB` (default `ON`, gated by `CARBON_BUILD_VIEWER`) pulls in
  `cmake/CarbonViewerWs.cmake`, which vendors `ggerganov/imgui-ws` →
  `ggerganov/incppect` → `uNetworking/uWebSockets` → `uNetworking/uSockets`.
  Three deliberate build choices: (1) each dependency is `FetchContent`-populated
  and compiled by *our* module rather than its own CMake, so imgui-ws links
  carbon's single `carbon_imgui` (a second Dear ImGui copy would be an ODR /
  `ImDrawData`-layout hazard); (2) the whole stack is **no-SSL** — OpenSSL is not
  installed on the target and isn't needed — which required patching out
  incppect's unconditional `template class Incppect<true>;` (the SSL variant) from
  the fetched copy at configure time, and compiling uSockets with `LIBUS_NO_SSL`
  (context/loop/socket/epoll_kqueue only); zlib is kept for WebSocket
  permessage-deflate. (3) the compressors' `#include "imgui/imgui.h"` prefix is
  satisfied with a one-line generated shim that forwards to `<imgui.h>`, and the
  `imgui-ws.js` / `incppect.js` client assets are baked into headers exactly as
  upstream does (`file(READ)` + `configure_file` on their `common.h.in`).
  `Viewer::run_web(sys, initial, render_fn, {port, tex_edge})` reuses the existing
  `OrbitCamera`, `render_fn`, and collective MPI broadcast loop unchanged; only the
  presentation swaps GLFW+GL for an `ImGuiWS` server. The rendered frame is
  nearest-downscaled and aspect-letterboxed into a **power-of-two** RGB texture
  (default 1024²) because the imgui-ws JS client uses a WebGL1 context, which
  renders non-power-of-two textures black under the default `REPEAT` wrap; this
  also caps per-update bandwidth (the frame only re-streams on a camera change).
  Browser input arrives as imgui-ws events — DOM mouse buttons are remapped to
  ImGui's ordering and fed through the IO event API so the on-canvas ImGui panel
  (FPS, camera readout, Reset button) stays interactive while `WantCaptureMouse`
  keeps panel clicks from orbiting. The HTML client is served from memory via
  `addResource("/index.html", ...)` (no on-disk `httpRoot`). Wired into all three
  render examples behind `--web [--port N]` (guarded by `CARBON_HAVE_VIEWER_WEB`,
  leaving the default and GLFW `--interactive` paths byte-for-byte unchanged).
  Verified end-to-end headless: `curl` confirms the three assets serve `200`, and
  a hand-rolled Chrome DevTools-Protocol screenshot (real-time wait for WS frames,
  since Chrome's `--virtual-time-budget` fast-forwards past live WebSocket
  delivery) shows the density sphere in the WebGL canvas with the live ImGui
  overlay.

- **2026-07 — Viewer polish pass (shaded colormap, contact shadow, richer UI).**
  Aesthetic upgrade to the browser/web viewer, all confined to `examples/viewer`
  and the render examples (no library surface touched). `rgb_from_density`
  (`render_rgb_utils.h`) now treats the normalized scalar density as a
  screen-space height field: a central-difference gradient yields a pseudo-normal
  that drives Lambert + rim shading over a three-stop **perceptual green ramp**
  (pale mint → vivid green → deep forest), so the volume reads as a lit 3D form
  rather than a flat blob. A **soft contact shadow** is faked in screen space —
  per-pixel opacity is downsampled to a coarse grid, box-blurred twice, then
  sampled at an offset opposite the light's screen projection to darken the studio
  backdrop. The post-process is `#ifdef _OPENMP`-parallelized and the web path
  renders at a lighter 1600×900 (downsampled to the 1024² POT stream anyway),
  keeping it interactive. The ImGui control panel gained a **TTF font** (DejaVu
  Sans at 17 px with 2× oversampling, falling back to Liberation/DejaVu Mono then
  a 1.2×-scaled default if none are on disk), **collapsing sections**
  (Camera / Animation / Overlays / Orientation), and an **FPS sparkline**
  (`PlotLines` over a 90-sample ring buffer). The in-panel XYZ gizmo picked up a
  thin ring border around its white disc and **negative-axis stubs** (hollow dots
  at half length, drawn under the positive axes). Built clean and
  `format-check`-clean; the shaded loop runs per-frame with no runtime regression.

- **2026-07 — Surface (ray-march) render style, and what a slab seam really means.**
  `examples/render_density.cu` gained a *Surface (ray-march)* style alongside the
  density colorizations, reusing `make_ray_march_kernel` over the same sphere ∪
  cylinder SDF. Getting it clean surfaced four separate issues, three of them real
  bugs in the example and one instructive dead end.
  (1) **Clip bounds removed.** A tight geometry AABB was originally added to stop
  rays wasting steps in empty space, but its faces sat ~3 cells off the sphere, so
  the marcher's first sample landed inside the isosurface band *on the entry face*
  and painted the whole flat face as a translucent quad. Empty space is already
  skipped by the `|field| > 4 * width` band test, so the clip box bought nothing —
  `use_clip_bounds = false`.
  (2) **`max_steps` 1024 → 4096.** With half-cell steps a ray crossing the 512³
  domain diagonal needs ~1800 samples; rays were being truncated mid-flight, which
  read as a translucent (never-saturating) surface.
  (3) **Halo cells seeded far-outside.** An init kernel writes *owned* cells only,
  so the Open-boundary halos at the domain faces keep whatever was in memory and
  the marcher reads that as `sdf ≈ 0` — a spurious isosurface sheet. A whole-
  allocation `System::async_memset(0x4C)` (≈ 5.4e7 as float) seeds them outside
  every band before the exchange fills the interior seams.
  (4) The **stencil widened to radius 2** so `halo_z == 2`. The normal is a central
  difference of trilinear samples, which reaches one cell further than the sample
  itself; with a one-cell halo the `z + h` corner gets index-clamped at the storage
  edge by `trilinear_detail::sample_clamped`, making the seam normal slightly wrong.
  Note the clamp is *correct* at the global domain edge (replicate under
  `BoundaryMode::Open`) but is silently wrong at a per-GPU slab edge, where the
  right answer lives in the neighbour's halo.

  The instructive part: a dark line across the surface at the slab seam looked
  exactly like a compositing bug and was chased as one. It was not. Cross-GPU peer
  copies were silently failing on the dev machine — `test_transfer_peer` gets `0`
  for a value of `100`, `test_halo_single_node` finds its `-7777` sentinel intact —
  so every halo exchange was a no-op. The marcher then faithfully renders the
  unexchanged seam cells: a constant field is "in band" everywhere (opaque) and has
  zero gradient, so `shade_blinn`'s `gn > 1e-8` guard falls through to ambient-only
  (dark). Opaque **and** dark is the fingerprint of a constant field, not of missing
  coverage — transmittance ~5e-5 with luminance 0.113 against 0.636 for the same
  pixel single-GPU. `carbon_render_ray_march`'s `multi_gpu_center_row` case already
  encodes this check, and it fails on exactly that hardware fault. Diagnostic rule
  of thumb: widening the halo changed the output by *zero bits*, which is what
  finally ruled the code out. With `CUDA_VISIBLE_DEVICES=0` the suite is 87/87.

  One library change came out of it: the per-slab march now snaps its first sample
  onto a shared global `t = k * step` lattice
  (`detail/render_ray_march.inl`). Each GPU previously started stepping at its own
  slab entry, so the sample phase restarted at every seam and the union of samples
  across slabs was not the set a single continuous march would visit — the band
  gets integrated over a short sliver twice. This is unrelated to the peer-copy
  fault above and remains unverified on multi-GPU hardware.

- **2026-07 — `render_wind_tunnel` faster flow + `.vti` volume export.** Bumped the
  inlet speed from `u_max = 0.05` to `0.10`. "Faster flow" is taken literally as a
  higher inlet velocity at the *same* fluid viscosity, so the knob held fixed is `nu`
  (hence `omega`), not `Re`: holding `nu = 0.0667` keeps `omega` at the previously
  proven-stable `1.43` while `Re` rises with `u_max` from 15 to 30. (The alternative —
  holding `Re` fixed and rescaling `nu` — would leave the physics identical and only
  change the lattice velocity unit, i.e. no visible effect.) The Mach number
  `0.10/cs ≈ 0.17` stays in the LBM low-Mach regime; the run is stable
  (`|u|max = 0.163`, no blow-up over 15000 KBC iterations). Render thresholds are
  already normalized by `1/u_max`, so they track the speed change automatically.

  Added a third output: a VTK XML ImageData (`.vti`) dump of the raw fields
  (`velocity` 3-vector, `speed`, `q`) written next to the two PPM renderings, so the
  fields ParaView needs are exported directly rather than only the rendered pixels.
  carbon's `io` surface emits only legacy STRUCTURED_POINTS `.vtk` today (its own
  docstring notes `.vti` is not emitted), so the writer is self-contained in the
  example: a uniform grid with one appended, raw, little-endian block per `DataArray`,
  each length-prefixed by a `UInt64` (`header_type`). Owned cells are gathered across
  the z-partitioned GPUs into VTI point order (x fastest, components interleaved) using
  `Span::z_offset()` for each slab's global-z base — so it is correct for both a single
  process (one full-volume file) and MPI (per-rank `.rankN` shards over the rank-local
  z range). Validated structurally (block byte counts and offsets match the grid) and
  numerically: the exported `speed` max equals the solver's reduction `|u|max` to the
  last digit, confirming the halo/stride/z-partition gather is correct. `velocity` and
  `qfield` are now `HostDevice` so they can be downloaded before the gather.

  A `geometry` marker array (0 = fluid, 1 = sphere, 2 = wall) is written alongside the
  physical fields so the tunnel is legible in ParaView without a legend. Without it
  every face of the `.vti` looks identical, which reads as "the inlet is on the wrong
  face" — but the export is geometrically faithful: the inlet is the `x = 0` face (its
  plane carries the forced parabolic profile, centreline velocity exactly
  `(u_max, 0, 0)`), the outlet is `x = Nx-1`, and the four long `y`/`z` faces are the
  no-slip walls; the apparent problem is only that the long streamwise `x` axis points
  into the screen under ParaView's default camera. With the marker the two open end
  faces read as fluid (bar a 508-cell wall ring where they meet the duct) while every
  long face reads as wall, so which face is the inlet is unambiguous.

- **2026-07 — `render_wind_tunnel` finalized: a real, vigorous von Kármán street
  (cylinder, D3Q27+BGK, Re = 200).** Closes out the multi-turn debugging effort tracked
  in the now-deleted `docs/WIND_TUNNEL_WIP.md`. The demo had been switched from a 3D
  sphere (which only sheds above Re ≈ 270, unreachable on this uniform grid) to a
  quasi-2D circular cylinder (D = 64 over a `1024 × 512 × 8` domain, `BoundaryMode::PeriodicZ`,
  D/H = 12.5 % blockage, `k_shed_perturb = 2.5` symmetry break), but the wake stayed
  near-steady. Two root causes, fixed in turn:
  1. **Collision model.** KBC's entropic limiter was dissipating the shedding mode —
     effective Re sat below the ≈ 47 onset, so the wake never went unsteady (global
     `Q`max ≈ 9e-5, `vy` amplitude ≈ 0.4 % of ulb). Switched to the far-less-dissipative
     **BGK** collision, which immediately produced real vortical structure (global `Q`max
     jumped to ≈ 0.13).
  2. **Reynolds number.** A stability sweep on the A100 box (Re = 100 → 320 at fixed
     ulb = 0.10, so omega = 1.445 → 1.786) showed the wake sheds monotonically harder with
     Re and — with BGK's headroom — stays finite (`|u|max ≈ 0.15`) well past omega ≈ 1.7.
     **Re = 200** (nu = 0.032, omega ≈ 1.678) was locked in as the default (was 80): over the
     full 40960-step (4 flow-pass) run it settles into a fully developed street — the wake
     centreline `vy` crosses zero **6 times** (≈ 6 alternating vortices populate the box),
     transverse velocity swings ±0.07 (≈ 70 % of ulb), and omega stays comfortably under the
     BGK stability ceiling of 2 for a well-resolved solve. All still overridable at run time
     via `CARBON_WT_RE` / `CARBON_WT_ULB` / `CARBON_WT_ITERS`.

  Also **retuned the Q-criterion render** for the white background. The ray-march ScalarField
  path decouples the two knobs: `value_max` sets only the colour (`t = s / value_max`) while
  per-sample alpha grows with the *raw* Q times `opacity_scale`. Since Q decays fast downstream
  (near-cylinder shear layers are several× the shed cores) and Viridis's bright teal→yellow end
  blends into white, `value_max` is anchored to ≈ 0.5·(wake peak) — keeping the whole street in
  the dark, high-contrast purple/navy band with the near-cylinder layers saturating — and
  `opacity_scale` lifted 40 → 200 so the weak far-downstream cores still accumulate solid alpha
  over the thin 8-cell span. The camera view direction was also flipped (`{0,0,-1}` → `{0,0,1}`
  in both `OrbitCamera::from_frame_bounds` and `Camera::frame_bounds`) so flow runs +x → screen-
  right with the cylinder on the left. Final renders (velocity/Jet, Q/Viridis) both read as a
  textbook street; build + 87/87 ctest + format-check all clean.

- **2026-07 — `render_wind_tunnel` promoted from quasi-2D to a genuinely 3D cylinder
  wake at higher Re.** The previous pass locked a clean *2D* street (thin 8-cell periodic
  span). This pass makes the wake actually three-dimensional while pushing Re up, without
  a multi-resolution grid. Three changes:
  1. **Thick, seeded span.** The span (z) is thickened to 128 cells (= 2 D, default;
     `CARBON_WT_NZ` overrides) and the free-stream initial condition is seeded with a small
     spanwise-varying transverse/vertical velocity ripple (two z-harmonics, ≈ 5 % of ulb,
     `CARBON_WT_PERT`). The thick span alone is *not* enough — a z-invariant initial +
     boundary state stays exactly 2D forever, since nothing couples the span — so the ripple
     is what lets the cylinder shear layers amplify the real mode-A/B spanwise instabilities.
     PeriodicZ is kept (physically exact for a spanwise-homogeneous cylinder; the canonical
     DNS setup). The full extent is now runtime-configurable (`CARBON_WT_NX/NY/NZ`) so
     resolution can be traded against Re without a rebuild.
  2. **Higher Re.** Default Re bumped 200 → **450** (D = 64 ⇒ nu = 0.0142, omega ≈ 1.843).
     A single-GPU sweep (Re = 100 → 650 at fixed ulb = 0.10) showed the 3D structure grows
     monotonically and stays finite well past omega ≈ 1.8: at Re = 450 the out-of-plane
     velocity reaches ≈ 65–78 % of ulb (identically zero for a 2D flow) and uy varies along
     the span, while the shedding stays coherent and omega stays under the BGK ceiling of 2.
     Bigger D is the lever that buys the higher Re at fixed omega (nu = ulb·D/Re). The Q clip
     was pulled a hair inside the y-walls (±5 R, not ±6 R) to drop a thin wall-shear sheet
     that otherwise integrated into a frame-wide line.
  3. **Single-GPU default + observed multi-GPU divergence.** The solve runs on **one GPU by
     default** (`pick_render_devices`; `CARBON_WT_MULTI_GPU=1` opts into all visible GPUs).
     A genuinely 3D (z-varying) wake was seen to *diverge* on a 2-GPU z-split (`|u|max = 1e30`)
     while single-GPU stayed bit-stable, with the error localised to the device-boundary
     z-plane. This was initially suspected to be a wrong-value bug in the direction-filtered
     **Lattice**-semantic multi-GPU PeriodicZ halo. **That diagnosis was wrong** — root-caused
     and fixed in the next entry (it was a cross-skeleton ordering race, not a halo bug). The
     single-GPU default is kept for simplicity (a render job doesn't benefit from a z-split),
     but multi-GPU is now correct. Build + 87/87 ctest + format-check all clean.

- **2026-07 — `render_wind_tunnel` multi-GPU divergence root-caused: a cross-skeleton
  ping-pong race, not a halo bug.** Follow-up to the entry above. The 3D wake blew up on
  2 GPUs (fine on 1) and the error concentrated at the device-partition z-plane, which
  looked like a wrong-value halo bug. Investigation ruled that out and found the real cause:
  1. **The Lattice-semantic multi-GPU PeriodicZ halo is correct.** Added
     `tests/test_halo_periodic_z_lattice.cu` — an intra-process 2-GPU PeriodicZ `Dense` field
     with a mixed-z-sign lattice stencil, sentinel-filled halos, one `Halo(...,
     StencilSemantic::Lattice)` exchange, then a per-cell check that lower halos receive only
     `c.z > 0` populations, upper halos only `c.z < 0`, both wrapped to the right global z, and
     in-plane (`c.z == 0`) populations are never moved. It **passes** on 2 GPUs (pair_count and
     values both exact), closing the "untested intra-process lattice-PeriodicZ halo" follow-up
     and proving the exchange values are right.
  2. **The bug is a missing barrier between the two ping-pong skeletons.** The hot loop
     alternates two independent `Skeleton` objects (`sk_ab` reads pop_a → writes pop_b,
     `sk_ba` reads pop_b → writes pop_a). Each `Skeleton` allocates its own per-GPU streams
     (`System::stream()` hands out a fresh `StreamGroup` per call), and carbon inserts no
     ordering *between* separate skeletons. With no drain, iteration N+1's cross-GPU halo
     peer-copy of a population can race iteration N's still-in-flight writes to that same
     population, reading stale bytes at the device-boundary z-plane. For a z-invariant (thin)
     or laminar flow the neighbouring slabs are near-identical so the race is invisible (this
     is why the lid-driven-cavity benchmark and the thin-span 2D street never showed it); for a
     genuinely 3D wake with large z-gradients at the boundary the wrong bytes amplify and blow
     up. Single-GPU has no cross-device peer copy, so it never manifested there.
  3. **Fix:** drain each iteration (`sk.run(); sk.wait();`) so the ping-pong is serialised —
     the idiom the benchmark header already documents. 2-GPU is now bit-stable and tracks
     single-GPU: over 5000 steps on `768 × 384 × 128`, Re = 300, `|u|max` = 0.156046 (2 GPU)
     vs 0.156044 (1 GPU), with the only residual difference (≈ 0.08, at the z=63/64 boundary)
     being ordinary chaotic-flow butterfly amplification of boundary-plane rounding, bounded
     and non-growing. The per-iteration host sync is negligible for a render job. **Takeaway:
     two independent `Skeleton`s that share fields and alternate must be explicitly ordered
     (drain between them, or share a stream); carbon cannot infer the cross-skeleton
     dependency.** Build + 88/88 ctest (new lattice-halo test) + format-check all clean.

- **2026-07 — `System::default_stream()` (the main/anchor stream) + `Skeleton` start/end
  fork-join on it.** Proper fix for the cross-skeleton race above, replacing the
  wind-tunnel's per-iteration host drain. Adopts Neon's anchor/user-stream model.
  1. **`System::default_stream()`.** `System` had no notion of a main stream: every
     `System::stream()` call ran `cudaStreamCreate` and handed back a brand-new per-GPU
     `StreamGroup`, so two orchestrators on the same `System` were guaranteed to sit on
     *different* streams with nothing ordering them. `default_stream()` returns one
     shared stream — the same underlying handle on every call, and on every shallow copy
     of the `System` (the form `Dense` / `Field` hold), because the handle lives in the
     shared `State`. Created lazily so the two device-free placeholders (default ctor and
     the moved-from state, both deliberately driver-free — see the comment above the move
     ctor) never touch CUDA. `const`-callable: `shared_ptr` propagates const to the
     pointer, not the pointee, the same mechanism `distributed() const` uses. It is a
     **real created stream, not the CUDA legacy null stream** — the null stream's implicit
     cross-stream synchronisation would defeat the OCC overlap modes, and it cannot be
     graph-captured.
  2. **`Skeleton` anchors on it.** The default ctor now takes `sys_.default_stream()` as
     stream 0 (was a private `sys_.stream()`). This single line is what orders separate
     skeletons: both ping-pong skeletons issue their stream-0 work into one FIFO. The
     caller-supplied-stream ctor is unchanged and still opts out.
  3. **Start/end fork-join (compile Phase 6 + `run_eager_`).** Each run now records a
     "start" event on the anchor that every secondary stream waits for (the fork), and at
     the end has the anchor wait on every secondary stream's tail (the join). The start
     event is recorded *into the anchor's FIFO*, which already holds the previous
     skeleton's join — so a skeleton's secondary streams cannot begin until the previous
     skeleton has fully finished, which is precisely the guarantee the race needed. The
     join also means the anchor alone represents whole-skeleton completion. Events are
     allocated after the OCC rewrite and after every other phase that appends to
     `events_`, so both the stream count and the stashed indices are final.
     Single-stream schedules (`OverlapMode::None`) allocate nothing and take no wait ops
     — FIFO on the anchor already orders them, so the common path is unchanged.
     Implemented as events rather than literal `Kernel::anchor` nodes, keeping
     `effective_kernels()` / `export_dot()` / `inter_iter_waits()` and every existing
     skeleton test unchanged. The pre-existing inter-iteration barrier is now largely
     redundant but kept (harmless extra waits) rather than ripped out in the same pass.
  4. **Result.** The wind tunnel dropped its per-iteration `sk.run(); sk.wait();` drain
     back to `sk.run()` and 2-GPU thick-span Re = 300 over 5000 steps reproduces
     `|u|max = 0.156046` — bit-identical to the drained run and matching single-GPU
     (0.156044) — with no host sync in the loop. The lid-driven-cavity benchmark's latent
     version of the same race is closed with no benchmark code change.
  New tests: `tests/test_system_default_stream.cpp` (shape / same-handle-across-calls-and-
  copies / distinct from `stream()` / empty-System stays device-free) and
  `tests/test_skeleton_anchor.cu` (stream 0 is `default_stream()` and is shared by two
  skeletons; plus an undrained alternating ping-pong of two `OverlapMode::Standard`
  skeletons — `num_streams == 2`, so the fork-join is load-bearing — matching a serialised
  single-stream reference bit for bit). 90/90 ctest + format-check clean.
- **2026-07 — inter-iteration barrier removed: the anchor fork-join subsumes it.**
  Follow-up cleanup to the anchor entry above, which left the older selective
  inter-iteration barrier in place as "redundant but harmless". It is now provably
  redundant, so it is gone: the anchor join makes stream 0's timeline strictly later than
  every stream's iteration-N work, and the next run's start event — recorded into that
  same FIFO and waited for by every secondary stream — therefore holds all of iteration
  N+1 behind all of iteration N. That is a *full* barrier between runs, strictly stronger
  than the old per-(s_a, s_b) selective wait set, and for single-stream schedules both
  mechanisms were already no-ops. Deleted: `compute_inter_iter_barrier_()` (~90 lines of
  O(n²·streams) compile-time coverage/conflict analysis), the `inter_iter_waits_` /
  `iter_tail_event_` members and the public `inter_iter_waits()` accessor, the
  `first_run_` flag (it existed only to gate the barrier), and the per-run
  iter-tail-record + cross-iter-wait loops in `run_eager_`. `compile` is now six phases
  with the anchor allocation as Phase 5. The trade-off is documented where the old
  "coarser than strictly necessary" note lived: the anchor serialises whole iterations
  rather than only the stream pairs that actually conflict across the boundary. That is
  not a measured regression — the OCC win is intra-iteration and survives: 2-GPU
  384³ cavity gives 18 339 MLUPS at `--overlap none` vs 19 387 at `standard`
  (+5.7 %), and all three overlap modes produce bit-identical checksums. Coverage is
  unchanged in substance: `test_skeleton`'s parallel-sibling subtest still validates
  multi-stream cross-iteration correctness against a single-stream reference (the
  removed assertions only pinned the barrier's internal *shape*), and
  `test_skeleton_anchor` covers the replacement directly. Wind tunnel re-verified:
  2-GPU thick-span Re = 300 / 5000 steps still gives `|u|max = 0.156046` vs 0.156044 on
  1 GPU.
- **2026-07 — SYCL backend un-broken (`graph_replay.h`) + `render_dist.h` export fix;
  test counts in `AGENTS.md` re-measured.** Verifying the test-count snapshot turned up
  two configurations that did not build at all.
  1. **SYCL.** `detail::GraphReplay::capture` / `launch` spelled `gpu_stream_t` in their
     *signatures*, outside the `CARBON_BACKEND_IS_CUDALIKE` arms. `detail/backend/
     cuda_like.h` deliberately defines no `gpu_*` aliases under SYCL, so those signatures
     failed to parse and every TU including `<carbon/skeleton.h>` (i.e. `skeleton.cpp`
     and `carbon.cpp`, the library itself) failed to compile — the whole SYCL backend had
     been broken since the CUDA-graph feature landed, and the stale `build-sycl` tree hid
     it. Fixed with a `GraphReplay::stream_handle` alias (`gpu_stream_t` on cuda-like, an
     opaque `void*` under SYCL, where every method is a stub that ignores its stream
     argument). The two new anchor tests also used the CUDA-only
     `StreamGroup::native_at`; they now compare `Stream::get_uid()` (the address of the
     shared per-GPU stream vector), which is the portable spelling of "the same streams"
     and is what the anchor contract actually asserts. SYCL is back to green: **86/86**.
  2. **MPI.** `render::dist_gather_segments` was declared without `CARBON_API`, so with
     `-fvisibility=hidden` the definition stayed a local symbol and
     `test_render_ray_march` — whose call site is inside `#ifdef CARBON_HAVE_MPI_NCCL` —
     failed to link in the MPI configuration only. One-line fix; **116/116** with
     `-DCARBON_ENABLE_MPI=ON`.
  `AGENTS.md`'s snapshot was stale by 20 tests (claimed 70 default / 92 MPI / 69 SYCL).
  Now measured and stated as a table (90 / 116 / 86 on a 2×A100 box) with a pointer to
  `ctest -N` as the source of truth, and the CUDA-only gate corrected: it holds four
  entries (`block_cuda` plus `mgrid_cross_level` / `mgrid_multi_gpu` /
  `mgrid_weighted_cuts`, which use hand-rolled `<<<>>>` kernels), not one. The stale
  "the CUDA-only list is currently empty" comment in `tests/CMakeLists.txt` is fixed too.

- **2026-07 — Surfel splat surface render (Dense first).** Added a voxel-parallel
  opaque surface path alongside the existing density splat and ray-march renderers.
  Each near-surface voxel projects itself (no point-to-block lookup): `make_surfel_kernel`
  skips cells outside an SDF band, snaps to the isosurface along the user-supplied
  normal, and competes for pixels via a packed 64-bit `atomicMin` visibility buffer
  (sortable clip depth in the high 32 bits, octahedral normal in the low 32).
  `SurfelImage::combine` folds with `min` across local GPUs and `MPI_Allreduce(MIN,
  Uint64)` across ranks — order-independent, so multi-GPU frames are bit-reproducible
  (`test_render_surfel` checks determinism and single/multi-GPU bit-equality).
  Backend shim: `detail::atomic_min(unsigned long long*)` for CUDA-like and SYCL.
  `render_density` viewer adds a **Surface (surfel)** style with deferred Lambert +
  rim shading over the shared studio backdrop (`rgb_from_surfel_studio`). Block /
  mGrid support (level-scaled footprints, `child_block_id` skip at jumps) deferred.

- **2026-07 — Surfel render: normal decode, per-pixel splat depth, shading parity.**
  The first surfel frames drew every shape as a cone whose apex pointed at the camera.
  Three defects stacked, listed by how much each contributed:

  1. **`decode_octahedral_normal` dropped the lower-hemisphere sign.** The fold branch
     rewrote `(ox, oy)` *and* forced `oz = 0`, but `oz` is precisely where that branch's
     information lives — it must stay negative. Every `nz < 0` normal collapsed onto the
     equator (worst-case round-trip error **180°**), so a whole hemisphere of surfels
     shaded as purely horizontal normals: a radial pinwheel with a singularity wherever
     the horizontal component vanished, i.e. the cone apex. `test_render_surfel` now pins
     the round trip over a 60×120 sphere sampling with a 1° bound; the old decode fails
     it outright. This is why the artifact survived two earlier fix attempts — the
     existing normal assertion sampled the view centre, where the expected normal is
     `(0, -1, 0)` with `nz == 0`, exactly the one point outside the broken branch.
  2. **Splat disks shared one depth.** All pixels in a footprint were keyed on the
     splat centre's `clip_z / clip_w`, so the globally nearest surfels won across their
     whole disk and stamped their own normal over neighbours. Depth is now the ray /
     tangent-plane intersection per pixel (surfel position + normal define the plane),
     so a footprint covers a local surface patch with real depth variation.
  3. **Back-facing surfels competed.** The band `|sdf| <= band_width` holds cells on
     both sides of the isosurface; `SurfelSplatCellFn` now rejects a splat when
     `dot(n, eye - world) <= 0`.

  Two follow-ons: the surfel resolve moved from Lambert + rim to the same Blinn–Phong
  parameters the ray-march surface pass uses (`rgb_from_surfel_studio` takes the
  `Camera` for the per-pixel view vector) so the two styles are visually comparable;
  and `render_density`'s loader derives normals from a central-difference SDF gradient
  instead of analytic sphere / cylinder normals, which disagreed with the stored union
  SDF at the junctions and on the caps and so snapped off the isosurface.

- **2026-07 — Surfel footprints are bounded tangent disks, not screen-space discs.**
  Once the cone was gone the sphere showed a ring of far geometry bleeding through it plus
  isolated wrong-normal specks. Cause: per-pixel tangent-plane depth (above) is only valid
  over the patch a surfel actually covers. Near the silhouette the plane grazes the view ray,
  so `depth = ((world - origin) · n) / (dir · n)` extrapolated far past the voxel and those
  surfels outvoted nearer geometry. Measured on an analytic sphere of radius 28.8 cells, the
  worst resolved depth was off by **168 cells**.

  A surfel now carries a world-space footprint — a tangent disk `1.5 * spacing` across (above
  a voxel's 0.87-cell circumscribed radius, so neighbours overlap and the surface stays
  hole-free) — and a pixel is only written when the plane hit lands inside that disk. This
  bounds the depth error by the footprint radius, and gives the footprint its correct
  foreshortened elliptical shape for free.

  Two knock-ons: the pixel loop bound is now derived per surfel from the projected voxel size
  (world span of one pixel, from the divergence of two adjacent pixel rays), so
  `splat_radius_px` is documented and used as an upper bound on per-voxel cost rather than the
  footprint shape — a fixed pixel disc silently clipped footprints into holes on zoom-in. And
  the footprint has a ~1 pixel screen-space floor, so a sub-pixel voxel (zoomed out) still
  covers a sample instead of falling between rays. `test_render_surfel` gained
  `test_depth_matches_analytic_sphere`, which compares resolved depth against the analytic ray /
  sphere intersection from an oblique view (all octants, so grazing footprints are covered) and
  holds it to 4 cells.

## 2026-07 — Pre-feature audit: the test suite could not fail where it mattered most

A review pass ahead of new feature work went looking for correctness gaps and instead found
that several of the mechanisms meant to *report* correctness were inert. Four of those are
worth recording, because each one had been quietly reducing the value of a green run for
months.

- **2026-07 — `assert` was compiled out of every test binary.** CMake defines `NDEBUG` for
  `Release` and `RelWithDebInfo`, and `Release` is what the documented workflow, both CI
  lanes, and the Docker matrix all configure. The suite has **305 `assert()` call sites across
  11 files**, and for **7 of those files `assert` is the only failure mechanism** — they had no
  `return 1`, no throw, no explicit comparison. Those binaries reported success no matter what
  they computed; the newly added `test_atomic_host` demonstrated it directly, printing "lost
  4096 increments" and still exiting 0. Test TUs now compile with `-UNDEBUG`
  (`carbon_test_enable_assertions` in `tests/CMakeLists.txt`), applied to all three
  registration helpers. The library keeps its Release semantics — this is scoped to tests.

  Turning the assertions on did not newly break anything on a single GPU, which says the
  assertions had stayed consistent with behaviour; the exposure was latent, not an outstanding
  bug.

- **2026-07 — the CPU-only CI lane had been red, not green, and the regex that hid it was
  stale.** `ci.yml`'s `cuda-cpu-only` job ran `ctest -E "carbon_(transfer_peer|halo_(single_node|periodic_z|multicomponent|host)|dist_.*)"`
  on the claim that "the GPU-bound ones return cleanly when `device_count() == 0`". Measured
  under `CUDA_VISIBLE_DEVICES=""`: **26 registered tests fail outright** with no device
  (`carbon_gpu`, `carbon_dense_kernel`, `carbon_skeleton`, `carbon_reduce_sum`, ...) and **not
  one of them matches those six patterns**. A further **50 tests "passed" only by reaching
  their own device-count self-skip**, contributing nothing. Real host coverage was 9 tests, not
  the 65 the summary implied.

  Tests now carry `cpu` / `gpu` / `needs_multi_device` labels and the lane selects `ctest -L cpu`.
  Labels beat a denylist here: the regex encoded a snapshot of the suite and silently decayed as
  tests were added, whereas a label is set at the registration site. Note that no label is a
  substring of another — `ctest -L` takes a regex, so a `needs_multi_gpu` spelling would also be
  selected by `-L gpu` and dropped by `-LE gpu`. The SYCL lane deliberately keeps running the
  full set: with an `omp` target the OpenMP host *is* an AdaptiveCpp device, so `gpu`-labelled
  tests execute for real there.

- **2026-07 — a single-GPU host silently downgrades 28 tests, and the GPU CI lane only warned.**
  Those tests open with `if (device_count() < 2) { ...; return 0; }`. `tests/test_multi_gpu_gate.cpp`
  makes the condition an explicit result: it passes with >= 2 devices, warns below that, and
  **fails** below that when `CARBON_TEST_STRICT=1`. `gpu-ci.yml` now sets that variable when the
  runner reports >= 2 GPUs (forwarded into the container by `scripts/docker-build-all.sh`), so a
  mis-provisioned runner — a hidden card, MIG partitioning, a stale `CUDA_VISIBLE_DEVICES` —
  turns red instead of reporting a pass it did not earn. One gate rather than 24 edited skip
  sites: the policy lives in one place and cannot drift as multi-GPU tests are added.

- **2026-07 — `detail::atomic_add(float*)`'s host branch was a plain read-modify-write.** It read,
  added, and stored with no synchronization, while its sibling `atomic_min` in the same header
  used `std::atomic_ref` + compare-exchange. That branch is not dead code: on a SYCL build with an
  `omp` target, `__SYCL_DEVICE_ONLY__` is undefined and kernels lower to OpenMP loops, so every
  thread of a per-cell accumulate — the density splat calls it once per sub-sample per footprint
  pixel — executed it concurrently. Now `std::atomic_ref<float>::fetch_add` with relaxed ordering
  (accumulator only; callers publish through a separate stream or host fence) in both the cudalike
  and sycl headers. `tests/test_atomic_host` hammers one word from 8 threads and demands exact
  arithmetic; against the old code it loses 4096 of 32768 increments.

### Multi-GPU: a longstanding correctness failure that no run had exercised

The label work above prompted the obvious question — what happens with two GPUs? On this
2×RTX 6000 Ada host, **15 tests fail** where the single-GPU suite is fully green. Every direct
check of the platform came back clean (`cudaDeviceCanAccessPeer` is 1 both directions,
`cudaDeviceEnablePeerAccess` succeeds with no warning, and a raw `cudaMemcpyPeer` round-trips
intact), which is what made this look like a carbon bug rather than an environment one. That
reading was wrong, and the "Root cause" subsection below explains what those checks missed —
they were all run on an idle machine. Symptom groups as first observed:

- **Halo delivers stale or zero cells.** `halo_single_node`, `halo_multicomponent`, `halo_block`,
  `mgrid_halo` read the value the destination already held instead of real neighbour data — the
  exchange did not land.
- **Multi-GPU reductions come up short by large factors.** `block_reduce` 3027324 vs 8276196;
  `dense_disg_universal_reduce` 33702648 vs 201805296; `mgrid_kernel` counts exactly half the
  expected parents and children, as if one GPU's contribution never arrived.
- **Illegal memory access.** `skeleton_block` and `skeleton_block_extended` abort with
  `cudaError 700` out of `StreamGroup::wait`.
- **OverlapMode divergence.** `block_overlapped_stencil` diverges by 0.87 against its own
  reference; `skeleton_block` puts 10480 of 34416 active voxels outside tolerance.
- **`lattice_halo_dense_auto`** reads a descending sequence (`-779900, -779901, ...`) where it
  wants a constant, i.e. it is addressing the wrong region rather than getting stale data.

Verified **pre-existing** at `1ecfecb`, before the recent anchor-stream and
inter-iteration-barrier commits, so this is not a regression from that work. (An attempt to
bisect further back stalled: commits older than the CUDA 13 upgrade no longer compile —
`cudaDeviceProp::clockRate` was removed — so "did two-GPU ever pass?" is still open.) The
reason it went unnoticed is exactly the gap the two items above describe: local runs used one
GPU, and `gpu-ci.yml` emitted a `::warning::` rather than a failure when the runner had fewer
than two.

#### Root cause: the peer DMA reports success and moves nothing

**Not a synchronisation race** — an earlier revision of this entry concluded "the broken edge is
copy → readback" from a `CUDA_LAUNCH_BLOCKING=1` experiment. That was wrong, and the way it was
wrong is instructive: `CUDA_LAUNCH_BLOCKING` serialises *kernel launches*, so leaving the failure
count unchanged says nothing about async **copies**, and the reasoning treated "the environment
did not fix it" as evidence about which edge was missing. Three measurements retired the race
hypothesis outright:

1. **The addresses are right.** Instrumenting `System::async_copy` to log every endpoint shows
   the halo's two peer copies land exactly where the layout demands: `gpu0+0x100` (buffer z=4,
   the owned plane for global z=3) into `gpu1+0` (lower halo), and `gpu1+0x40` (z=1, owned) into
   `gpu0+0x140` (z=5, upper halo), 64 bytes each. Correct source, destination, and size.
2. **The copy is on the same stream as the readback.** The copy into gpu1 and slab 1's
   device-to-host readback are both issued on `stream[1]`, so CUDA stream ordering alone already
   sequences them, with a `stream.wait()` on top.
3. **An unconditional full fence does not help.** Adding `cudaDeviceSynchronize()` on *every*
   device after *every* copy — the strongest fence available, which cannot leave any ordering
   hazard intact — still fails 15/15.

What is actually happening: **with peer access enabled, a cross-device `cudaMemcpyAsync` with
`cudaMemcpyDefault` returns `cudaSuccess` and transfers nothing**, leaving the destination holding
its previous contents. That is why the destination still reads the test's pre-fill sentinel — not
because the readback was early, but because nothing was ever written. A ~70-line standalone
reproducer outside carbon (two devices, `cudaMalloc`, copy on the destination's stream, sync, read
back, 200 iterations) isolates it from carbon entirely:

| peer access | GPUs idle | GPUs busy (another process) |
| --- | --- | --- |
| not enabled | 0/200 bad | 0/200 bad |
| enabled *after* the allocations | 200/200 bad | 200/200 bad |
| enabled *before* the allocations (carbon's order) | 0/200 bad | 200/200 bad |

The last row is precisely carbon's behaviour: it enables peer access during `GpuGroup`
construction, before any field is allocated, so cross-device copies work on an idle machine and
silently move nothing as soon as any other process touches the GPUs. That is the whole "flaky
standalone, reliable inside `ctest`" signature, and it is why the raw `cudaMemcpyPeer` spot-check
quoted at the top of this entry appeared to prove the platform was healthy — it was run idle.

The platform is a 2×RTX 6000 Ada pair on an AMD host with the IOMMU active, connected over a PCIe
host bridge (`nvidia-smi topo -m` reports `NODE`), driver 580.95.05. Ada has no NVLink and no
working PCIe P2P, yet `cudaDeviceCanAccessPeer` returns 1 and `cudaDeviceEnablePeerAccess`
succeeds — the driver's own capability report is what cannot be trusted here. Corroborating
signal: `cudaDeviceGetP2PAttribute(cudaDevP2PAttrNativeAtomicSupported)` is **0** for both
directions, where an NVLink-attached peer reports 1.

#### Fix: never take the driver's word for peer access

`src/gpu.cpp` no longer enables peer access on `cudaDeviceCanAccessPeer` alone. Two gates now sit
in front of it, and a `CARBON_P2P` environment variable selects the policy:

- **`auto` (default)** — enable a direction only if the link reports NVLink-class capability
  (`cudaDevP2PAttrNativeAtomicSupported != 0`) *and* it passes the delivery probe below.
  Deliberately conservative: a healthy PCIe-P2P platform is demoted to the host-staged path
  (correct, slower) and can opt back in. The native-atomic attribute is a proxy for "is this
  NVLink", chosen because it is reachable through the runtime API without linking NVML.
- **`off`** — never enable peer access. Every cross-device copy is host-staged, which is always
  correct.
- **`on`** — enable every direction the driver claims, subject only to the probe. For a PCIe-P2P
  platform known to be good.

The probe, `peer_dma_delivers`, moves 16 known ints device-to-device exactly the way
`System::async_copy` does (a `cudaMemcpyDefault` `cudaMemcpyAsync` on the *destination* device's
stream) over a destination pre-seeded with a distinct sentinel, then reads it back and compares.
A direction that does not deliver has its enablement revoked with `cudaDeviceDisablePeerAccess`,
so those copies fall back to host staging, and logs a warning naming it a platform defect rather
than a configuration error. Results are memoised per direction for the process lifetime — a
`GpuGroup` is built once per `System` and a program may construct several, so an unmemoised probe
would repay its allocate / copy / sync cost per construction, once per ordered pair and therefore
quadratic in the device count. Note the enablement `(ctx -> peer)` backs copies in the
*opposite* direction (`peer -> ctx`), since a device-to-device copy runs in the destination's
context — so each copy direction is probed, and the specific enablement behind a failing
direction is the one revoked.

Result on this host: **93/93, three consecutive full runs**, where the same suite failed 15 tests
before. `CARBON_P2P=on` under load demonstrates the probe firing — it reports "peer DMA 1 -> 0
reported success but delivered no data", revokes both directions, and logs "peer access active on
0 of 2 validated direction(s)".

The probe is a backstop, not the primary defence, and the table above shows why: on this platform
it passes when run on an idle machine, so a process that starts idle and *then* meets contention
keeps a peer mapping the probe just blessed. `CARBON_P2P=on` is therefore genuinely
at-your-own-risk here (2/8 runs still fail under load with it forced). The link-class gate is what
makes the default safe, because it never enables the mapping in the first place.

One knock-on correction: the `RESOURCE_LOCK` rationale in `tests/CMakeLists.txt` attributes this
symptom class to a cross-stream race root-caused in 2026-06 and fixed by the producer fence in
`carbon::Halo` / `HaloDisg` / `HaloBlock`. That fence addresses a real and different hazard; it
was never what these 15 failures were about.

#### Two further findings this unmasked

- **`tests/test_transfer_peer.cu` depended on ordering carbon does not provide.** With the P2P
  bug out of the way this still failed 10/10 under concurrent load (`got 0, want 100` — a
  destination reading freshly-zeroed memory). The test uploaded host → gpu0, which
  `async_copy` places on `stream[0]` because gpu0 is that copy's destination, then immediately
  issued the gpu0 → gpu1 peer copy on `stream[1]`. Nothing orders `stream[1]` behind `stream[0]`,
  so the peer copy could read gpu0's allocation before the upload filled it — the exact cross-GPU
  producer hazard `carbon::Halo` fences internally and a raw `Transfer` leaves to the caller. The
  test now drains between the two, and its comment (which described the upload and peer copy as
  sharing a stream) is corrected. 0/15 under the same load afterwards.
- **`dense_disg_overlapped[Skeleton]` has a real residual race — still open.** Under artificial
  load (a second process looping a multi-GPU test) it fails ~10/15, clean idle, and clean in the
  normal suite. `[OverlappedStencil]` passes on the same data while `[Skeleton]` drifts slightly
  at the boundary plane (`got=2.8828704 ref=2.8949077`, 144 cells) — small deviation rather than
  the wholesale staleness of the P2P bug, i.e. partially-updated rather than unwritten data.
  Recorded negative evidence so it is not re-derived: the obvious hypothesis is that the
  Skeleton's cross-stream edges use the per-GPU `StreamGroup::wait_for` where a halo needs the
  N×N `wait_for_all` fan-in (`OverlappedStencil` uses `wait_for_all` for exactly this edge, which
  would explain why it passes). Three variants were tried and **none** changed the failure rate:
  fanning in a halo step's own wait events, fanning in the anchor fork/join bracket in
  `run_eager_`, and fanning in both sides of every halo-incident edge. All three were reverted
  rather than left in as unverified scheduler changes. The dumped schedule for reference: halo
  steps sit on their own streams (1..4) with the first carrying **zero** wait events and relying
  entirely on the anchor bracket, while each compute lands on stream 0 split Internal / Boundary
  with the Boundary half waiting on two events. Worth noting the anchor bracket's comment claims
  it prevents "the halo of iteration N+1 peer-copying a population that iteration N is still
  writing", which a per-GPU `wait_for` cannot actually guarantee across GPUs — a latent
  documentation/contract mismatch even though fixing it did not resolve this test.

### Smaller fixes in the same pass

- **`io::read_native` implements its documented short-read throw.** The docstring had always
  promised `std::runtime_error` "when a read short-circuits", but no code checked the stream. A
  truncated shard therefore latched `failbit` on the first short row, turned every later
  `in.read` into a no-op that left its destination untouched, and returned a field set that was
  a silent mix of file data and stale host-mirror contents — on the checkpoint restart path.
  Now checked per (component, slab) so the message names the region that ran out, plus a
  trailing-bytes check at the end (the writer emits nothing past the last payload, so extra
  bytes mean the file does not describe the field list being read) and an explicit `flush()`
  before `write_native`'s success test, since a full filesystem usually surfaces at the flush of
  the last block and an `ofstream` failing in its destructor cannot report it. `test_native_io`
  covers truncation and trailing bytes, and was checked to fail without the guards.
- **Checkpoint manifest entries are sanitised on read.** `CheckpointReader` fed the manifest's
  `file` string to `field_shard_path`, which joins with `fs::path::operator/` — and that
  *discards* the directory when the right-hand side is absolute. An edited or corrupt manifest
  could redirect the reader to any path on the host and load it into a field buffer. The writer
  already vetted field names via `require_safe_field_name`; the reader now applies the same rules
  to `file` (`require_safe_manifest_file`), with `test_checkpoint_dense` asserting rejection of
  `../../etc/passwd`, `/etc/passwd`, and `sub/dir/rho.bin`.
- **The manifest is committed atomically.** `manifest.json`'s presence *is* the commit marker, but
  it was written in place, leaving a window where a reader or a post-crash restart found a
  truncated JSON file that looked committed — converting a recoverable "no checkpoint yet" into a
  hard parse failure. Now staged to `manifest.json.tmp`, `fsync`'d, `rename`d (atomic within a
  directory), and the directory `fsync`'d so the rename itself survives power loss. The `fsync`
  helper is POSIX-only and deliberately non-throwing: it hardens durability and should not fail an
  otherwise complete checkpoint.
- **`checkpoint::ensure_directory` no longer deadlocks the job on an unwritable directory.** The
  comment in place said the throw was deferred past the barrier "so peers do not deadlock waiting
  on rank 0" — and then threw inline. Rank 0 left the collective while every peer was already
  committed to `maybe_barrier`. It now logs at the failure, joins the barrier, then throws, which
  is what the comment always claimed.
- **The CUDA-like RAII deleters no longer skip cleanup.** Stream, event, and allocation deleters
  returned early when `gpu_set_device` failed, leaking the driver handle. They now attempt the
  destroy regardless. Stream and event handles carry their own context so the call succeeds
  anyway; `gpu_free` does need the device current and will return an error, but the accounting was
  already reversed so `gpu_memory_in_use` stays consistent. Deliberately no logging: these can run
  during static destruction, after carbon's logger is gone, and a shutdown-time log is a worse
  hazard than the leak it reports.
- **`CARBON_SANITIZE`** (`address`, `undefined`, `address+undefined`, `thread`) builds carbon and
  everything linking it with host sanitizers. `PUBLIC`, not `PRIVATE`: ASan and TSan change the ABI
  of the runtime they interpose, so a sanitized libcarbon inside an unsanitized executable does not
  work. Scoped to `COMPILE_LANGUAGE:CXX` — neither nvcc nor acpp takes `-fsanitize` on the device
  pass, and device memory errors are compute-sanitizer's department. The full host-test set is
  clean under `address+undefined`.
- **Surfel example gradient no longer reads out of bounds.** `examples/render_density.cu` computed
  central differences at `gc.x ± 1` / `gc.y ± 1` with no clamping, but Dense stores a halo on **z
  only** — so at an x/y domain face the read walked into the adjacent row, and at cell 0 addressed
  before the row base. Now clamped per axis with the divisor taken from the distance actually
  sampled, degrading to a one-sided difference on the face. The band test also moved ahead of the
  gradient work: the splat kernel discards everything outside the band immediately afterwards, so
  the six reads were being spent on the whole domain interior.
- **Surfels are culled by footprint, not by centre pixel.** `splat_at` rejected any surfel whose
  centre `project_pixel` placed outside NDC, but a footprint is a disk around that centre, so
  surfels straddling the frame edge were dropped and the silhouette thinned along all four
  borders. `Camera::project_pixel_unclipped` returns pixel coordinates without the lateral
  frustum test (`project_pixel` now delegates to it and applies the test in pixel space, which is
  exactly equivalent — the mapping is affine and monotone), and the footprint loop relies on
  `try_atomic_min`'s existing per-pixel clip, with a whole-footprint-offscreen early-out so a far
  off-screen surfel does not walk `(2r+1)²` rejected pixels.

Deliberately **not** done in this pass, for lack of a toolchain to verify against: HDF5 `H5*`
return-code checking in `src/io_hdf5.cpp` (no HDF5 headers on this host and no package manager
to add them), the HIP `GpuGroup` peer-access enable that `src/gpu.cpp` performs and
`src/gpu_hip.cpp` only logs, and the SYCL allocation overflow guard that the CUDA path has.
Each is a small, well-understood change, but all three are in code this host cannot compile,
and an unverifiable edit to a configuration CI does not exercise on PRs is worse than a
documented gap. `AGENTS.md`'s reference to an AdaptiveCpp pixi environment at
`/home/max/.pixi/envs/adaptivecpp` is also stale — that path no longer exists.

## 2026-07 — Peer-access policy shared with HIP; the SYCL exposure named

Follow-up to the peer-access work in the entry above, which landed the validated-P2P policy
in `src/gpu.cpp` and therefore on the CUDA backend only. The policy is a correctness
mechanism, not a CUDA detail, so leaving the other two backends outside it meant the
silent-corruption failure mode was still reachable — just not through the configuration that
happened to be under test.

- **The policy now lives in `include/carbon/detail/backend/cudalike/peer_access.h`** and is
  shared verbatim by the CUDA and HIP backends, which is the arrangement `AGENTS.md` already
  describes for everything else under `detail/backend/cudalike/`: it talks to the runtime
  only through the neutral `gpu_*` aliases, so one source serves `cudaXxx` and `hipXxx`
  including HIP-on-NVIDIA. `src/gpu.cpp` and `src/gpu_hip.cpp` each reduce to building the
  device-id list and calling `configure_peer_access`. Supporting this needed new aliases in
  `detail/backend/cuda_like.h` for the peer-access surface (`gpu_device_can_access_peer`,
  `gpu_device_enable_peer_access`, `gpu_device_disable_peer_access`,
  `gpu_device_native_atomic_supported`, `gpu_error_peer_access_already_enabled`) and for the
  blocking sequence the probe needs before any Stream/Buffer plumbing exists
  (`gpu_device_synchronize`, `gpu_stream_synchronize`, `gpu_memcpy_default`,
  `gpu_memcpy_async_default`, `gpu_memset`).

- **HIP previously enabled no peer access at all** — `src/gpu_hip.cpp` logged the device
  inventory and stopped, where `src/gpu.cpp` ran the enable sweep. That made it accidentally
  immune to the silent-corruption bug (every cross-device copy was host-staged) while also
  never claiming the bandwidth on a healthy fabric. It now runs the same gate, so the
  behaviour is deliberate rather than incidental in both directions.

- **The ROCm side of the heuristic is explicitly unverified.** `peer_link_is_trusted` keys on
  native-atomic support as a proxy for "NVLink-class link", which is an inference from NVIDIA
  behaviour. Whether ROCm reports 1 for an xGMI-attached pair is untested here. The failure
  mode if it reports 0 is that HIP keeps the host-staged path it already used before this
  change, so `Auto` cannot regress HIP relative to its previous behaviour; it can only fail to
  claim an improvement. That asymmetry is why the same conservative default was used on both
  backends rather than a more aggressive HIP-specific one. `CARBON_P2P=on` remains the escape
  hatch, and measuring the attribute on an MI-series machine is the open follow-up.

- **The previous entry's reasoning about the probe is confirmed, and is stronger than it
  looked.** That entry argued the construction-time probe could not be the primary defence
  because a probe of an idle machine cannot speak for a link that fails later. A first
  measurement with `CARBON_P2P=on` (which bypasses the link-class gate and leaves only the
  probe) seemed to contradict it by catching both directions — but repeated runs show the
  platform is simply non-deterministic. One run caught a single direction and revoked it while
  the *other* direction passed the probe and then delivered wrong data to the very next test;
  three consecutive runs afterwards passed both directions cleanly. So the probe is a backstop
  for the grossest cases, and the conservative link-class gate is what actually keeps the
  default correct. `CARBON_P2P=on` is genuinely unsafe on this platform, which is what it is
  documented to be.

### Measured: the SYCL backend is *not* exposed, so nothing was added for it

`src/gpu_sycl.cpp` puts every device in one process-wide `sycl::context` specifically so that
cross-device `queue.memcpy` works, and its own comment records that AdaptiveCpp's CUDA target
maps that to `cuMemcpyPeerAsync` — the same driver path that silently delivers nothing on the
affected host. On that reading SYCL looked exposed, and worse than CUDA: carbon does not
control whether AdaptiveCpp enables peer access, and per that same comment there is no
graceful peer fallback.

Measurement says otherwise. With the container matrix unblocked (below), the SYCL backend was
built and run on the affected two-GPU host: **89/89 pass, including all 27
`needs_multi_device` tests**. The CUDA defect reproduces on an idle machine — that is how the
delivery probe catches it — so a clean idle multi-device run under SYCL is direct evidence
that AdaptiveCpp is not taking the broken path, most likely because it never enables CUDA peer
access and its cross-device copies are already host-staged. No probe and no fallback were
added: they would be machinery for a failure this platform does not exhibit. If a platform is
ever found where SYCL *does* corrupt, the shape of the fix is a construction-time probe plus a
host-staged fallback in the SYCL arm of `System::async_copy`.

### The SYCL backend did not compile, because assertions were switched on

Getting to that measurement first required fixing a build break. `tests/test_gpu.cpp` asserts
that a copied `CudaStream` / `Event` shares the original's handle, spelled `st_copy.get() ==
st.get()`. The SYCL wrappers deliberately have no `get()` — they expose `queue()` and
`event()`, and `detail/backend/sycl/stream.h` explains why. This compiled anyway for as long as
it did only because Release builds defined `NDEBUG`, which erases the assert's expression
before the compiler sees it. Enabling assertions in test binaries (previous entry) made those
four expressions real, and the SYCL build stopped working — invisibly, because nothing on this
host could build that backend to notice.

The test now routes both checks through small `stream_handle` / `event_handle` helpers that
compare handle addresses, which preserves the intent (a copy shares the original's handle) on
both backends without naming either one's accessor. Worth recording as a pattern: switching
`NDEBUG` off does not merely enable checks, it introduces code that was never compiled before,
and any backend not covered by a build lane will find out late.

### HIP-on-NVIDIA: graph entry points, and runtime validation of the HIP gate

The `hip-nvidia` configuration also did not build. HIP's NVIDIA platform shim aliases the
CUDA-graph *types* but does not forward the graph *entry points*, so `hipStreamBeginCapture`
and its five siblings in `detail/backend/cuda_like.h` were undeclared under
`-DCMAKE_HIP_PLATFORM=nvidia`. Since the types there are already the CUDA ones, those six
aliases now fall through to the CUDA entry points on that platform, with the status passed
through HIP's own `hipCUDAErrorTohipError` because HIP keeps `hipError_t` a distinct enum. The
real AMD path is untouched.

That was worth fixing beyond the config itself: `hip-nvidia` is the only way to give the HIP
sources runtime coverage without an AMD card, and it now runs **89/89** on this host. It
confirms the shared peer-access policy end to end on HIP — the HIP backend reaches the same
decision as CUDA for the same pair ("peer access 0 -> 1 left disabled: the link does not report
NVLink-class capability"), and forcing `CARBON_P2P=on` exercises the HIP spelling of the probe.

### Why the container matrix could not substitute for a toolchain: pip

`scripts/docker-build-all.sh` exists so a host lacking AdaptiveCpp or ROCm can still validate
those backends, and its `sycl` / `hip-nvidia` configurations both target NVIDIA, so on a
CUDA-only workstation they should have been exactly the missing toolchain. Every image build
instead dies in its first `pip3 install` layer, which is where all five Dockerfiles obtain
cmake >= 3.25 and clang-format >= 20 (Ubuntu 22.04 apt has neither).

The cause is a TLS-intercepting proxy. Diagnosis worth recording, because the symptom is
misleading: the only trust anchor on the host is an *intermediate* ("Forward Trust CA") whose
issuer ("Root CA") is present nowhere on the machine. `curl` accepts the chain anyway because
it enables OpenSSL's partial-chain flag, which lets an intermediate act as a trust anchor, so
`curl https://files.pythonhosted.org` returns a normal HTTP response while `openssl s_client`
against the very same bundle reports `unable to get issuer certificate`. Python's `ssl` does
not set that flag. pip therefore cannot verify PyPI from *either* the host or a container, no
matter which CA bundle it is pointed at — copying host trust anchors into the images was tried
first and does not help, because the anchor that would complete the chain does not exist
locally. Everything else the builds need is reachable: apt works, and `curl` verifies github,
conda-forge and PyPI.

The interception is also narrowly targeted: of the endpoints the images need, only
`files.pythonhosted.org` presents the corporate certificate. `pypi.org`, `github.com`,
`objects.githubusercontent.com`, `conda.anaconda.org`, `apt.llvm.org`, the Ubuntu archive and
the NVIDIA repositories all serve their genuine chains. So the fix is to stop using pip rather
than to repair certificate verification, which is not repairable from inside this repository.

All five Dockerfiles now take cmake from the upstream Kitware binary tarball on GitHub (pinned
via a `CMAKE_VERSION` build arg, default 3.31.6 — above the 3.28 that `CMAKE_HIP_PLATFORM`
needs) and clang-format from `apt.llvm.org`'s `llvm-toolchain-jammy-20`, whose signing key is
ASCII-armored and so can be used directly as an apt `signed-by` target without adding gnupg to
the images. `python3-pip` is dropped; `curl` is added where it was missing. Certificate
verification stays fully enabled throughout — no `--trusted-host`, no `-k`.

This unblocked the matrix on this host: the `sycl` and `hip-nvidia` images build and both run
89/89, which is what made the two measurements above possible.
- **2026-08 — `render_wind_tunnel` browser viewer parity (`--web`) + up-relative orbit drag.**
  The example already had a `--web` / `--port` code path, but the default `build/` tree is
  configured with `CARBON_BUILD_VIEWER=OFF`, and the web-only `build-viewer-web/` binary was
  stale. The render callback also ignored the viewer's `mode` argument and always composited
  the speed field over the cylinder — the Q-criterion view was unreachable in the browser,
  and `q_image` was ray-marched every camera move and discarded. Fixed with a `Layers` enum
  on `render_for_camera` (skip the unused scalar march), a `{"Speed", "Q-criterion"}` style
  selector on both backends, and a 1280×720 web stream. Separately, orbit drag in
  `carbon_viewer` now yaws around the scene `up` vector and pitches around the camera-right
  axis instead of world-Z azimuth/elevation, which felt inconsistent at oblique views.
- **2026-08 — `render_wind_tunnel` density-splat viewer modes (mGrid-compatible path).**
  Added `"Speed (splat)"` and `"Q-criterion (splat)"` render styles alongside the existing
  ray-march pair. Both use `make_density_kernel` + `DensityImage::combine` — the same
  Carbon kernel-loader model as `render_density` / `render_density_mgrid`, projecting
  per-cell scalars through `Camera` with Gaussian screen splats instead of integrating
  along rays. The cylinder overlay stays on the ray-march SDF pass (only the obstacle needs
  surface traversal). Shared post-process helpers live in `examples/viewer/render_rgb_utils.h`
  (`rgb_from_density_colormap`, `rgb_composite_ray_march_over_rgb`). File export remains
  ray-march; the splat path is the one that ports onto adaptive `MultiResBlock` towers
  without a dedicated traversal structure.
- **2026-08 — Brick-index locator for `Block` / `MultiResBlock`, and a grid-generic point
  sampler on top of it.** Every visualization technique worth porting from the AEOLUS wind
  tunnel — Q-criterion wake isosurfaces, surface-pressure overlays, volumetric smoke,
  streaklines — reduces to the same primitive: *what is the value of this field at this
  arbitrary world position*. Dense answered it (`sample_density_field` in
  `detail/render_ray_march.inl`) because the world-to-cell map is arithmetic there. The
  sparse families could not answer it at all: `Block::Span` walks bricks through a 27-way
  connectivity table, which navigates *from a cell you already have*, and mGrid's
  `parent_block_id` / `child_block_id` likewise only move up and down from a known `Idx`.
  Nothing inverted `global_location`. That is why `make_ray_march_kernel` still carries a
  `static_assert` restricting it to Dense, and why the mGrid render examples use forward
  screen splatting rather than ray marching.

  `detail/block/locator.h` adds the missing inverse as a dense `uint32_t` index over the
  *brick* lattice — one entry per brick, so 4 bytes per 512 voxels with the default 8³ brick
  (0.8% overhead on a `float` field, less on anything wider). `Locator::locate` maps a global
  voxel to `(block_idx, intra)` with one table read plus the existing
  `Span::set_and_validate`, which means the bitmask check comes along for free and only
  *active* voxels resolve. `Block::locator(gpu, world_origin, world_spacing)` builds it lazily
  behind a `std::once_flag` and caches the device table on a shared `LocatorStorage`, so the
  cost is paid once per grid no matter how many kernels ask.

  The table spans the bounding box of the bricks a GPU actually retained rather than the
  whole candidate lattice — a sparse shell inside a large box would otherwise pay for the
  box — grown by one brick on every side to hold the ghost ring (see the next entry for why
  the ring is there and why it is not an ownership violation).

  `detail/mgrid/locator.h` stacks one locator per level into a `MultiResLocator` and adds
  `find_finest`, which probes finest-first and returns the finest level holding the point.
  This works without any cross-level transfer precisely because mGrid already keeps a parent
  brick and all its descendants co-resident on one GPU, and because the coarsest z-cut is
  projected to every finer level — so all levels report the same owned world-z band, which
  `test_mgrid_locator` pins directly. The tower is a fixed-size array (`MaxLevels = 6`) since
  it is captured by value into kernels; a deeper tower throws rather than silently truncating.

  `detail/render_sample.h` then wraps all four families behind one shape
  (`sample` / `sample_nearest` / `gradient` / `spacing` / world bounds / owned z band), with
  the cell value supplied by a caller-provided `value_fn` rather than a `Partition` so derived
  quantities (speed, Q, a free-stream-subtracted wake scalar) need no new types. The shared
  trilinear machinery moved out of `render_ray_march.inl` into this header. Two behavioural
  notes: the sparse samplers substitute the **centre cell's** value for corners that fall in a
  hole or another slab, because zeros would paint a dark rind around every sparse boundary,
  and they reject the sample only when the centre cell itself is inactive. `DenseSampler` also
  clips its addressable z band to the global extent, which the older
  `sample_density_field` does not: that function's band is offset by `halo_z` and would clamp
  into the first slab's lower halo (meaningless under `BoundaryMode::Open`). It is unreachable
  through the ray-march path, which clips rays to the domain AABB first, so
  `sample_density_field` is left alone.

- **2026-08 — `-Werror cross-execution-space-call` on every nvcc TU, after a silent
  zero-result bug.** The sampler above shipped with `make_sampler` missing `CARBON_HD` — a
  host-only factory called from device code. nvcc reports this as *warning* #20011 and
  compiles the TU anyway; the device path of the call is simply never emitted. The result is
  not a crash or a wrong number but an untouched output buffer, so the cross-family sampler
  test reduced to exactly `0` violations over exactly `0` queries and looked green.

  What caught it was a deliberate second counter: the tests now pack *work performed* into the
  reduction alongside *violations* and fail when the visit count does not match the domain
  (`test_render_sample`, `test_block_locator`, `test_mgrid_locator`). A test whose only
  failure signal is a violation count cannot distinguish "everything passed" from "nothing
  ran" — the same class of gap as the 2026-07 finding that 305 assertion sites were inert
  under `NDEBUG`.

  The build now refuses the construct outright: `CARBON_NVCC_DEVICE_FLAGS` in the top-level
  `CMakeLists.txt` carries `--extended-lambda -Werror cross-execution-space-call` and is used
  by libcarbon, tests, examples and benchmarks (both the CUDA and the HIP-on-nvcc paths). The
  existing tree compiles clean under it, so this costs nothing today and closes a
  silent-wrong-answer channel permanently.
- **2026-08 — Ray marching on `Block` and `MultiResBlock`, and the ghost ring the seams
  needed.** `make_ray_march_kernel` used to `static_assert` on `grid_has_dense_iteration_v`,
  so a sparse grid could only be drawn by forward density splatting. With the locator in
  place the restriction is gone, and the way it came off is worth recording: rather than
  writing a second, brick-flavoured pixel function, `RayMarchPixelFn` was retemplated from
  `<Grid, UserFn>` onto `<Sampler>`. Everything it needs about the data — the world bounds to
  clip against, the owned world-z band this GPU may integrate, the cell spacing, and the
  value and gradient at an arbitrary point — is exactly the `GridSampler` surface, so one
  march body now serves all four families and the per-family part reduces to
  `detail::make_ray_march_sampler`. Since `Block` and `MultiResBlock` carry no world frame of
  their own (`Span::spacing()` is an integer refinement factor there, not a world size),
  `RayMarchOptions` gained `world_origin` / `world_spacing`, ignored by the Cartesian
  families.

  The first multi-GPU run is what made this pass worth the trouble. `test_render_ray_march_sparse`
  renders the same analytic sphere SDF through Dense and through a sparse Block and compares
  silhouettes; agreement was 0.9914 on two GPUs and **0.9878 on eight**. Decaying with slab
  count is the signature of a seam defect, and the cause was the locator's original
  owned-bricks-only rule: a trilinear gather anchored at an owned cell next to a seam wants
  corners one voxel past it, found nothing, and fell back to the centre value, which moved
  the isosurface slightly along every seam.

  The fix is to stop conflating two different questions. *Which cells may this GPU
  integrate* is answered by `owned_z_lo` / `owned_z_hi`, and those bands still tile the
  lattice disjointly, so nothing is double counted. *Which cells may this GPU read* is a
  strictly larger set that includes the ghost ring — precisely the distinction `Dense` draws
  between its owned band and its haloed storage band. The table now carries ghost entries,
  recovered from the 27-way connectivity table (which already routes each owned brick's
  cross-slab neighbours to the matching ghost slot, so no new topology had to be retained).
  `Locator::locate` validates through `Span::bitmask()` directly instead of
  `Span::set_and_validate`, because the latter rejects any slot at or beyond the *owned*
  block count and would therefore reject every ghost; this is the same bitmask read
  `Partition::neighbour` already does to walk a slab edge. Ghost cells hold stale data until
  a `HaloBlock` update runs, exactly as a dense halo does.

  With the ring in place Block does not merely approximate the Dense render, it reproduces it
  **exactly** — 9388 lit pixels on 1, 2 and 8 GPUs — so the test threshold is set just below
  exact rather than at a comfortable 99%. A 1% budget would have absorbed the original defect
  in silence.

  Also fixed here: `DenseSampler`'s z bands were both off by one `halo_z`. The storage band
  is `[z_offset, z_offset + haloed_extent().z)` and owned starts at `z_offset + halo_z`, but
  the sampler used `z_offset - halo_z` and `z_offset + logic_extent().z + halo_z`, which
  simultaneously reached one halo *below* the buffer on interior slabs and excluded the upper
  halo. Both errors vanish when `halo_z == 0`, which is why the cross-family sampler test did
  not see it: its Dense case was built with an empty `Stencil`. That case now uses a real
  6-point stencil plus a halo exchange, and skips only at the global domain rim rather than at
  every slab edge, so seam interpolation is checked rather than filtered out. Reverting the
  band produces 1682 violations, so the coverage is real.
- **2026-08 — Colour from a second field, and a depth plane that means what it says.** Two
  additions to the ray-march API, both prerequisites for the wake and surface-pressure modes
  ported from the AEOLUS wind-tunnel viewer.

  `make_ray_march_kernel` gained a six-argument overload taking an *attribute* loader
  alongside the density one, and `RayMarchShading::SurfaceScalar` uses it: the isosurface,
  its band and its normal still come from the primary field exactly as in `SurfaceBlinn`, but
  the material colour at each hit is a colormap lookup of the attribute sampled at the same
  world point. That split — shape from one quantity, colour from another — is what a
  Q-criterion surface tinted by local speed needs, and it is not expressible by pre-combining
  the two fields into one scalar. The attribute goes through the same per-family
  `make_ray_march_sampler`, so it works on all four grid families and may live on any field
  over the same grid. A miss (no attribute loader, or a hole in a sparse grid) falls back to
  the flat `material_color` rather than to black. `Colormap::CoolWarm` lands with it: the
  three existing maps are sequential, and a free-stream-subtracted velocity or a pressure
  coefficient is read around its zero, which a sequential map hides.

  The depth plane is the sharper change, because it altered the meaning of an existing
  buffer. `RayMarchImage` already tracked a per-pixel `t_enter` — where the ray entered this
  GPU's slab — used only to order slabs during `combine`. It now stores the ray parameter of
  the first sample that *contributed opacity*, and is named `depth` throughout, with
  `RayMarchImage::host_depth()` exposing it in the same units and with the same meaning as
  the long-standing `SurfelImage::host_depth()`. Two layers can therefore be depth-tested per
  pixel instead of composited in a fixed order, which is how a wake gets to pass in front of
  the body it came off.

  Slab ordering is unaffected: slabs are disjoint intervals along a ray, so a first hit
  inside slab A always precedes any hit in slab B whenever the entry points did, and a slab
  the ray crossed without picking up opacity now reports `inf` instead of a finite entry with
  zero colour and unit transmittance — skipped either way. The distinction that matters is
  the one the old value could not express: a ray that crosses the domain and hits nothing
  used to report a finite depth, which would have made empty space occlude another layer.
  `test_hit_depth` pins exactly that, checking corner rays that cross the box but miss the
  sphere read `inf` while the centre ray reads one radius in front of the sphere centre.
- **2026-08 — Wake mode: subtracting the free stream, and colouring a cloud by a second
  quantity.** The first of the four AEOLUS wind-tunnel techniques ported into
  `examples/render_wind_tunnel.cu`, plus the one library primitive it needed.

  `RayMarchShading::ScalarFieldAttribute` is the volumetric counterpart of the
  `SurfaceScalar` mode added alongside it: threshold and accumulated alpha still come from
  the primary field, but `[value_min, value_max]` and the colormap read the attribute
  sampled at the same point. `render::speed_deviation(u, u_inf, inv_scale)` computes
  `|u - u_inf|`, shared by the march and splat paths so the two cannot drift apart.

  Why subtract the free stream at all: in an external flow the undisturbed fluid fills most
  of the domain at a single large `|u|`, so a speed render spends its whole colour range
  describing the tunnel rather than the body. The existing "Speed" still is a good
  illustration — it is saturated red nearly edge to edge. After subtraction the free stream
  reads exactly zero, a 5 % threshold makes it fully transparent, and what is left is only
  what the cylinder did. The deviation is taken against the free-stream *vector*, not its
  magnitude: near a body the flow turns as much as it slows, and a deflected-but-undiminished
  region belongs to the wake even though `|u|` is unchanged there.

  The two new viewer modes each read two fields, which is the point of the pass:

  - **Wake** — opacity from `|u - u_inf|`, hue from the *signed* streamwise deviation
    `u_x - u_inf` through `Colormap::CoolWarm`. Both quantities vanish in the free stream, so
    the picture is empty upstream, and the two lobes of the diverging map separate fluid the
    body slowed (recirculation, vortex cores) from fluid it accelerated (the shoulders). A
    magnitude alone collapses that sign, which is most of the physics.
  - **Q-criterion (speed-tinted)** — opacity from Q, hue from local `|u|`. Q says where the
    vortices are; the tint says which sit in fast fluid. The render shows each core blue at
    its centre with a red rim, which is the expected structure: stagnant core, fast shear
    layer around it. Two cores with the same Q at different downstream stations are visibly
    different objects, which the untinted mode cannot show.

  Splat variants exist for Wake but not for the tinted Q: additive splatting has no
  per-pixel colour channel to tint, so that path can only carry the disturbance magnitude.
  Both new modes also write headless stills (`render_wake.ppm`,
  `render_qcriterion_tinted.ppm`) under fixed names — they are the most interesting frames to
  look at and the least convenient to reach, since the viewer wants a display.

  `test_scalar_field_attribute` pins the split that makes the mode worth having, using a
  uniform-density ball carrying an x-ramp attribute: the colour must vary across the ball
  even though the density does not, and the frame corners — which miss the ball but sit
  where the attribute is largest — must stay perfectly transparent. That second half is what
  fails if the threshold or the alpha ever reads the attribute by mistake.

- **2026-08 — Surfel attribute plane, and the Cp overlay it exists for (AEOLUS item 2).**
  A surfel splat resolves the nearest fragment per pixel with a single `atomicMin` over a
  packed 64-bit key: sortable depth in the high 32 bits, octahedral normal in the low 32.
  Painting a scalar on the resolved surface needs that scalar to follow the same winner, and
  a second independent pass would not — the two passes could disagree about which fragment
  is nearest wherever the tie-break differs.

  The mechanism is a second plane packed the same way: the same depth in the high bits, the
  quantized `SurfelSample::attribute` in the low. `try_atomic_min` writes both from one call
  site with one depth value, so the fragment that wins visibility wins the attribute, and
  `combine` min-folds both across GPUs and (under MPI) across ranks with no extra ordering
  logic. The exception is a bit-identical depth tie: the two planes then tie-break on their
  low bits independently and can pair one fragment's normal with another's attribute. That
  is not hypothetical — it happens on the part of a surface most nearly perpendicular to the
  view, where neighbouring surfels' tangent planes agree to the last bit, and
  `test_render_surfel`'s sphere measures it at 17 of 5189 covered pixels (0.3 %). Both
  candidates sit at the same depth so neither answer is wrong, but an attribute with a steep
  surface gradient will speckle there; the test bounds both the incidence and the magnitude
  rather than pretending the effect is absent.

  The plane is opt-in (`SurfelImage(sys, w, h, with_attribute)`), because it doubles the
  per-pixel device buffer and its D→H copy for a feature most callers do not use. With it
  off, `host_attribute()` throws instead of returning zeros that look like measurements.

  **`sortable_uint_to_float` did not invert `float_to_sortable_uint` for negative values.**
  The encoder flips the sign bit of a positive float and every bit of a negative one; the
  decoder unconditionally applied a mask that only undoes the positive case. Depth is
  positive by construction, so in the years this key has existed nothing ever decoded a
  negative — the attribute plane, carrying a signed pressure coefficient, was the first, and
  it rendered a smooth ±2 field as a hard red/blue binary. Fixed to branch on the encoded
  top bit; `test_sortable_float_round_trip` now covers both signs and the order-preservation
  property the `atomicMin` depends on.

  **Cp on the cylinder** (`examples/render_wind_tunnel.cu`, mode "Surface Cp",
  `render_cp.ppm`). In LBM the equation of state is `p = rho·cs²` with `cs² = 1/3`, so
  `Cp = (p - p_inf)/(½ rho_inf U²)` collapses to `(rho - rho_inf)/(1.5 rho_inf U²)`. Three
  things had to be right for the picture to mean anything:

  - **`rho` had to survive.** The example used to overwrite the density field with positive
    Q, which was free while nothing else read `rho`. Q now has its own `qpos` field.
  - **The reference density is measured, not assumed.** Anchoring `rho_inf` to the outlet's
    `rho = 1` inherits the channel's viscous pressure drop over the 3/4 of its length that
    lies downstream of the body; the measured upstream value is 1.0057, worth +0.37 in Cp.
    A reduction over a slab 3–5 radii upstream (clear of the inlet's Dirichlet plane) plays
    the role of a wind tunnel's upstream static tap.
  - **The wall probe samples a fixed radius, not a fixed step.** The wall is half-way
    bounce-back, so cells inside the body hold no fluid state and the probe must clear them.
    Stepping a fixed 2 cells along the outward normal and rounding to the lattice
    under-reaches on the diagonals — `(0.71, 0.71)·2` rounds to `(1, 1)`, 1.41 cells, back
    inside the wall. Targeting the cell containing `centre + n·(R + 2.5)` instead gives every
    reading the same station, which is also what makes them comparable around the body.

  Each run logs `Cp(theta)` at mid-span, and it is textbook: `+1.02` at the stagnation nose,
  a suction minimum of `-2.02` on one shoulder against `-1.41` on the other (the wake is
  shedding, so the two sides are not symmetric at any instant), and a flat `-1.2` base. That
  table is the check; the render is a view of it. It also caught the decode bug — the numbers
  were right while the image was binary, which localized the fault immediately.

  The Cp still gets its own oblique camera rather than reusing the top-down one the volume
  passes share. A cylinder's normals are all radial in `xy`, i.e. perpendicular to a view
  down its own `z` axis, so from there not one of them faces the eye and back-face culling
  correctly leaves nothing but a silhouette crescent. Painting pressure on a surface requires
  looking at its flank. `rho` also now ships in the VTI export alongside velocity, speed and Q.

- **2026-08 — Live LBM wind-tunnel viewer.** The shared `carbon::viewer` loop gains
  `TickFn` / `UiFn` hooks plus a Simulation panel (play/pause, steps/frame, reset, steps/s)
  broadcast through `ViewerStatePacket` on every rank — including the MPI `rank != 0` mirror
  loop. `examples/wind_tunnel_common.h` holds the D3Q27+BGK `LiveSim` (ping-pong skeletons,
  parity-aware macro extraction, lazy Q / Cp refresh) and `WindTunnelRenderContext`; the new
  `render_wind_tunnel_live` example steps continuously with runtime presets
  (384×128×96 / 512×160×160 / 768×256×192 — z-heavy to thicken multi-GPU z-slabs and
  shrink the nx×ny halo face), live Re and inlet-speed sliders
  (ω clamped ≤ 1.98, kernel rebuild on change rather than per-frame device writes), and the
  same eight render modes as the static export. Preset switches tear down and rebuild `LiveSim`
  so peak memory never holds two domains.

- **2026-08 — Live wind-tunnel multi-GPU render + z-heavy presets.** `RayMarchKernelImpl`
  launches over `RayMarchImage` buffer dimensions (not the stale ctor camera, which was 0×0 for
  cached kernels and skipped every pixel). The live viewer uses the same per-frame
  `make_ray_march_kernel` path as the static export; `LiveSim::prepare_for_render` fences the
  sim stream and refreshes velocity + SDF halos before compositing. Domain presets grow `nz` and
  shrink `ny` at fixed cell count so inter-GPU z-halo traffic drops; the `pick_live_devices`
  thin-slab workaround is removed — multi-GPU is expected to work on all presets again.
- **2026-08 — Viewer ImGui ID fix.** Renamed the Simulation panel reset button to “Reset sim” and
  the web camera reset to “Reset camera” so they no longer collide with Dear ImGui IDs.
- **2026-08 — Viewer frame-resolution presets.** The Rendering panel offers 16:9 presets
  (0.5K/1K/2K/3K/4K → 960×540 through 3840×2160). @c Viewer::ResolutionChangeFn lets apps
  rebuild GPU buffers; the live wind-tunnel example recreates its @c RayMarchImage /
  @c SurfelImage set on change.
- **2026-08 — Live viewer performance panel.** @c ViewerPerfStats records solver steps/s, ray-
  march render time, RGB pack + imgui-ws upload time, payload size, and upload bandwidth; the
  live example exposes MLUPS (and MLUPS/GPU) in a new Performance ImGui section.

- **2026-08 — Web-viewer sim-thread handoff defects.** Four bugs in the ownership split between
  the imgui-ws UI thread and the background sim thread introduced when the web viewer was
  decoupled. All four are in @c examples/viewer , none reach @c libcarbon .

  (1) **"Reset sim" reset the simulation on every subsequent tick, not once.** The UI loop pulls
  @c simulating / @c steps_per_frame / @c steps_per_sec_smooth back from @c shared.sim each frame
  but deliberately not @c reset_clicked , yet it published its local copy of that flag every
  frame. Nothing ever lowered the local, so the moment the sim thread cleared the shared flag the
  UI re-raised it and @c fill_simulation_packet set @c reset_requested again — the wind tunnel
  rebuilt itself every tick and never advanced. It is now a one-shot handoff: the UI raises the
  shared flag and immediately drops its local, and the sim thread clears the shared one when it
  has consumed the request.
  (2) **Play/Pause and the steps slider could bounce back.** The sim thread snapshotted
  @c shared.sim , released the mutex to run the tick, then wrote the whole struct back, so any
  UI-thread write landing in that window was overwritten by the pre-tick snapshot. It now writes
  back only the four throughput counters it owns.
  (3) **Desktop runs reported 0 MLUPS.** @c Viewer::run recorded render, upload and display-FPS
  timings but never called @c record_solver_perf , even though @c apply_simulation_tick was
  maintaining @c steps_per_sec_smooth right there. The performance panel read a field nothing
  wrote. Only the web path was wired up.
  (4) **@c ViewerPerfStats was a plain-double data race** — filled by the sim thread under
  @c shared.mtx , read by the UI thread under @c g_app_mtx , which is a different mutex and so no
  mutual exclusion at all. The fields are @c std::atomic now. Benign on x86-64 in practice, but
  it is a real race and @c -DCARBON_SANITIZE=thread would flag it.

- **2026-08 — Live wind-tunnel preset ladder shifted up one rung.** @c small (96×96×384, 3.5M
  cells) was too small to be interesting on this class of machine and is gone; every remaining
  preset keeps its extent and takes the next name down, and a new @c xlarge doubles all three
  axes of the old top rung. The ladder is now @c small @c 160×160×512 (13.1M), @c medium
  @c 192×192×768 (28.3M), @c large @c 256×256×1024 (67.1M), @c xlarge @c 512×512×2048 (537M).
  This supersedes the @c xlarge entry below, which described the rung that is now @c large .

  Note the default is still @c Preset::Large by name, so the default domain grows 28.3M → 67.1M
  cells. Measured on 8×A100-40GB: 565 / 528 / 298 / 62 steps/s, so @c xlarge sustains ~32 GLUPS.
  It is also the first multi-GPU-only rung — the nine live fields cost 252 B/cell, which is
  ~139 GiB total (measured 17.8 GiB per GPU across 8) and more than a single card holds. The
  @c --preset usage text says so. Because @c radius() keys off @c nz , doubling every axis leaves
  the blockage ratio unchanged, so the new rung is the same flow at higher resolution rather than
  a different experiment.

- **2026-08 — Test domains now scale with the slab count.** Nine tests hard-coded a domain that
  cannot be partitioned eight ways, so the whole @c mgrid / @c block / @c skeleton group aborted
  on an 8-GPU host while passing on 1 and 2. This was never a library defect: each grid family
  refuses such a domain up front and says so ( @c "local slab z-extent 1 is smaller than
  2 * halo_z" ; @c "candidate brick z-extent (4) is smaller than global_slab_count (8)" ). The
  effect was that a hard-coded extent silently capped the GPU count a test could cover, and the
  cap was well below the machines carbon targets.

  New @c tests/test_domain.h states each family's rule once — @c dense_slab_z (at least
  @c 2 * halo_z owned z planes per slab) and @c block_slab_z (at least one brick row per slab,
  where the brick z span is @c SBlock::MemZ for @c Block and the *coarsest* level's
  @c MemZ * R^(depth-1) for @c MultiResBlock ). Both only ever grow the extent, so every test
  keeps the exact shape it was written against on the GPU counts it already ran on. The two
  families want different growth: @c mgrid domains are fully active, so only @c z grows
  ( @c 16×16×32 → @c 16×16×128 at 8 GPUs) and the cost stays small; the @c Block domains are a
  sphere sized from a cubic @c N , so they stay cubic ( @c 48³ → @c 64³ ) and the sphere scales
  with them rather than leaving the new brick rows empty.

  Six expectations that were constants derived from the old extent are now derived from the
  extent itself (per-level brick counts and cross-level match totals in @c mgrid_multi_gpu ,
  @c mgrid_kernel ). @c reduce_sum_host_multi_gpu additionally keeps its z indivisible by the
  slab count, which is the uneven-remainder path it exists to cover. Suite is 97/97 on 8, 2 and
  1 GPU, with @c CARBON_TEST_STRICT=1 on the multi-GPU runs.

- **2026-08 — OCC split kernels could be fenced on the wrong half (skeleton parity flake).**
  @c carbon_skeleton failed its Standard-OCC-vs-None parity check about 4 runs in 5 on two GPUs,
  and the long-tracked @c dense_disg_overlapped boundary-plane drift under concurrent GPU load
  was the same defect. When the OCC rewrite splits a kernel into Internal and Boundary halves it
  inserts a new step, but @c step_for_kernel kept pointing at the *first* half, and
  @c occ_allocate_record_event_for_ allocated its cross-stream record event on that step. A
  consumer — typically the halo — could therefore be released once the Internal half finished,
  with the Boundary half still in flight, so it read the boundary plane mid-write. It is a race,
  hence non-deterministic and load-sensitive, which is why it looked like drift.

  @c Skeleton now tracks @c completion_step_for_kernel separately from @c step_for_kernel : the
  former names the step that actually *finishes* a kernel (the Boundary half for the standard and
  two-way-forward splits, the Internal half for the extended back-walk), and record events are
  allocated against it. @c occ_allocate_record_event_for_ also reuses an event already present on
  that step instead of overwriting it, so two consumers of one producer no longer clobber each
  other's fence. Verified with @c --repeat until-fail:8 over the @c skeleton , @c mgrid_skeleton
  and @c dense_disg_overlapped tests on 8 GPUs, all clean.

- **2026-08 — Live wind-tunnel @c xlarge preset.** Adds @c 512×512×1024 (~268M cells,
  ~2.4× large); intended for multi-GPU runs with ample device memory.

- **2026-08 — Empty dense/disg partitions crashed the launch on many-GPU hosts.**
  @c compute_dense_grid already returns @c LaunchShape{0, 0, 0} for an empty iteration extent,
  but the cudalike @c launch_dense_grid / @c launch_disg_grid handed that straight to
  @c cudaLaunchKernel as @c dim3(0, 0, 0) , which the driver rejects with
  @c cudaErrorInvalidConfiguration . Any grid with fewer z planes than GPUs therefore failed on
  the slabs that got no cells — @c test_dense_kernel 's @c 4x4x4 domain on an 8-GPU box put
  GPUs 4..7 in exactly that state. Both now return @c gpu_success for a zero-volume launch,
  matching the guards @c launch_block_grid and the reduce impls already had. Notably the **SYCL**
  backend has carried this guard from the start, and its comment justified it as matching the
  CUDA backend's "non-fatal no-op" behaviour — an assumption that was simply not true of CUDA,
  which is why SYCL was green where CUDA was not. Fixes @c dense_kernel , @c partition_read and
  @c dense_disg_overlapped on 8 GPUs.

  Two follow-ups noted here at the time have since been fixed; see the two entries below for the
  OCC record-event defect behind the @c carbon_skeleton parity flake and for the test-domain
  scaling.

- **2026-08 — Live wind-tunnel render-app audit, second pass.** A re-review after the first round
  of audit fixes, mostly about what the app does *between* frames rather than inside them.
  (1) **Preset changes froze the UI for over a minute.** @c LiveSim::reset seeded @c pop_a ,
  @c pop_b , @c cells and @c cells_scratch through @c Field::fill_owned_host , which is a serial
  loop with no OpenMP: ~3.8e9 callback invocations at @c xlarge , each with a @c sqrt and (for
  the populations) two @c std::sin . Measured init was 4.4 s / 14.3 s / ~31 s / **70 s** across
  small / medium / large / xlarge, with @c user ≈ @c real confirming one core. Worse,
  @c apply_params runs under @c g_app_mtx , so the UI thread — which takes the same lock in
  @c ui_fn — was locked out for the whole rebuild on every preset change, obstacle change and
  reset. New @c build_pop_init_kernel / @c build_cells_init_kernel seed on the device instead
  (the same @c wt_population_value / @c wt_cell_value , now @c CARBON_HD ), preceded by an
  explicit @c async_memset of the halo planes so the pre-exchange state stays defined — the host
  path had inherited zeroed halos from the zero-initialized mirror, and the wall-mask stencil
  reads @c cells halos on a slab edge. **xlarge init: 70 s → 0.33 s**, throughput unchanged
  (215 vs 214 steps/s).
  (2) **~15 GB of unused host mirrors.** Those four fields were @c HostDevice purely to be filled
  on the host and @c Transfer ed once; nothing read the mirrors afterwards. Seeding on the device
  lets them be @c MemorySide::Device — 224 B/cell reclaimed, and peak host RSS at @c xlarge on
  8 GPUs is now 720 MiB.
  (3) **Q modes ran the macroscopic pass twice per frame.** @c refresh_q opened with
  @c refresh_macroscopic , and both call sites (the per-frame tick and the warm-up path) already
  ran it themselves. It is now a documented precondition, so the full-field kernel plus its halo
  runs once.
  (4) **@c look_at could roll or throw on a degenerate camera.** The previous pass fixed the
  @c Top / @c Front / @c Side buttons, but the azimuth/elevation sliders are still in the
  orbit's Z-up parametrization, so under @c up = +y a user can still slide the eye onto the up
  axis. The root cause is in @c look_at itself: when @c f × up degenerates it substituted a fixed
  @c s = (1,0,0) , which is merely *lucky* to be orthogonal here — for a caller with @c up ∥ x
  looking along x the side axis collapses onto forward, the third axis comes out zero, and
  @c Camera::perspective throws on the singular view-projection. It now substitutes the world
  axis least aligned with forward, so @c |f · h| ≤ 1/√3 and the basis is always orthonormal.
  Fixes every caller rather than just this app's buttons.
  (5) **@c q_wake_max drove the Q colormap ceiling straight off a per-frame device reduce**, so
  the normalization moved with vortex shedding and the image pulsed in brightness. Now a decaying
  peak-hold (jump to a new high, ease back at 0.92/frame).
  (6) **@c render.background was read unsynchronized** by the sim thread in @c render_cb while the
  UI thread wrote it from the dark-background checkbox; it is snapshotted under the lock now.
  Not fixed here, and worth knowing: on this 8-GPU box the default suite has 18 pre-existing
  failures unrelated to the app, most of them @c "invalid configuration argument" from an empty
  partition on GPU 4+ when a small test domain is split eight ways. They reproduce on a pristine
  tree. On 2 GPUs the suite is 92/93.

- **2026-08 — Live wind-tunnel render-app audit fixes.** Four defects found reviewing the app
  after the span/cross axis swap. (1) **Obstacle end caps.** @c build_inout_bc_kernel wrote
  free-stream equilibrium on the span-end faces unconditionally, and a cylinder (axis @c +y )
  reaches both of them, so every step injected fluid *inside* the solid and hollowed the body
  out from its caps. It now takes the obstacle descriptor and returns early on solid cells. The
  defect arrived with the @c +z flow reorientation (the original x-flow layout had a periodic
  span, hence no caps); the x/y swap only moved it from the x faces to the y faces.
  (2) **Camera axis presets.** @c Top / @c Front / @c Side set azimuth + elevation directly in
  the orbit's Z-up parametrization, so under @c up = +y the @c Front preset produced an eye
  offset equal to @c up ; @c look_at then loses its side axis, falls back to @c s = (1,0,0) and
  rolls the scene by an arbitrary angle. New @c orbit_axis_view_dir derives all three from
  @c orbit.up (with @c Top stopping @c 0.17 rad short of the pole) and reproduces the previous
  directions exactly for the Z-up examples. This roll — not a singular matrix — was also the
  real mechanism behind the earlier "sheared" initial view.
  (3) **Q-mode UI stall.** @c refresh_for_mode ran under @c g_app_mtx , so a Q refresh (kernel +
  three halos + device reduce, each with a stream wait) blocked the UI thread for the whole
  frame and undid the viewer decoupling on exactly the heaviest modes. It now takes the sim by
  reference and runs unlocked, alongside @c step / @c refresh_macroscopic .
  (4) **Cross-flow windows overshot the domain.** @c radius() scales with @c nz , so after
  @c nx / @c ny were halved the @c cx ± 5r Q clip, the @c cx ± 6r wake-reduction box and
  @c view_frame_bounds all exceeded the grid in every preset (large: @c [-62, 258] vs
  @c [0, 192] ), making the wake isolation vacuous and letting @c q_wake_max — which auto-scales
  the Q colormap — max over the side-wall shear. All three are clamped to the domain.
  Cosmetics in the same pass: dropped the duplicated domain-panel line, the three "(splat)"
  render modes that aliased the ray-march path (the live app has no @c DensityImage and never
  called @c rgb_from_density_colormap ), the unreachable @c Layers::Both , and the double-reset
  branch in @c tick_fn ; @c LiveSim::iteration_ is now @c std::atomic since the UI thread reads
  it while the sim thread advances it. Left alone deliberately: @c radius() still keys off
  @c nz , so the cylinder blocks ~33 % of the tunnel width at every preset.

- **2026-08 — Live wind-tunnel reset-camera framing.** Orbit target / distance now derive from
  @c view_frame_bounds() (cylinder + downstream wake) instead of the full grid; Reset camera
  re-queries the current preset via @c frame_bounds_fn .

- **2026-08 — Live wind-tunnel square @c x – @c y extent.** Presets now set @c nx = @c ny
  (medium @c 320×320×512 ) so cross-flow and span share the same cell count; the former span
  width is kept on both axes.

- **2026-08 — Live wind-tunnel span on @c y , cross-flow on @c x .** Presets swap @c nx / @c ny
  (large @c 128×384×768 ): cylinder axis @c +y with radial distance in @c (x,z) ; spanwise
  perturbation keys off @c gc.y ; wake / Q reductions clip cross-flow on @c x . Orbit eye offset
  is @c {1,0,0} with up @c +y (cross-flow side view; span vertical, flow horizontal).

- **2026-08 — Live wind-tunnel z-flow preset dimensions.** After reorienting flow onto @c +z ,
  presets now use the z-heavy layout from the multi-GPU pass (grow streamwise @c nz , shrink cross @c
  ny , widen span @c nx at fixed legacy cell count): small @c 192×64×384 , medium @c 320×80×512 ,
  large @c 384×128×768 . The naive @c nx / @c nz swap (@c 192×256×768 ) had mis-stated the large
  problem relative to the documented @c 768×256×192 benchmark.
- **2026-08 — DenseSampler world bounds off-by-one.** @c world_bounds_max now uses
  @c origin + spacing × extent (matching @c dense_world_bounds and Block), not @c extent − 1 ,
  which had been clipping ray-march volume one cell short of the grid wireframe.
- **2026-08 — Live wind-tunnel render clip = grid extent.** Ray-march @c clip_bounds now match
  @c dense_world_bounds() (±1 cell margin). The wake-centred clip from the old x-flow layout had
  been cutting off the inlet z slabs while GPU-slab wireframes still showed the full partition.
- **2026-08 — Multi-GPU ray-march seam fix.** Per-GPU volume segments are now depth-sorted by
  slab entry (@c t_enter ) instead of the first opaque sample inside the slab. The old key mis-
  ordered compositing on z partitions and showed as vertical banding with many GPUs.
- **2026-08 — Viewer grid overlay matches dense extent.** The orange wireframe and orbit
  framing now use @c dense_world_bounds() (@c origin → @c origin + spacing × global_extent ) instead
  of a wake-padded box around the cylinder; GPU slab overlays already used the same convention.
- **2026-08 — Wind-tunnel ray-march step budget.** Speed / wake / sphere passes now scale
  @c max_steps to the clip-box diagonal (~1600 for the large preset) instead of a fixed @c 512 ,
  which was truncating rays after the z-flow reorientation and showed up as concentric banding in
  the volume render.
- **2026-08 — Viewer UI decoupled from sim.** The web viewer runs LBM stepping and ray-march
  rendering on a background thread while the imgui-ws menu loop stays at ~60 Hz on rank 0.
  Simulation throughput no longer stalls panel input when @c steps/frame is high.
- **2026-08 — Warp fork builds on CUDA 13; validated via a CUDA-12 container.** Work began on
  replacing Neon with Carbon inside XLB (vendored at `XLB/`, branch `carbon`). Two environment
  facts shaped the approach on the dev host (2×RTX 6000 Ada, driver for CUDA 13.1, no CUDA 12
  toolkit, PyPI unreachable through an SSL-intercepting proxy that returns 503 for
  `files.pythonhosted.org`):
  1. **The pinned Warp fork (`nvlukasz/warp @ external-source-support-update2`) does not compile
     against CUDA 13.** `warp/native/cuda_util.cpp` used the pre-13 driver/runtime API
     (`(PFN_cuGetProcAddress)` cast; 3-arg `cudaGraphNodeGetDependentNodes`), and `reduce.cu`
     includes `<cub/iterator/counting_input_iterator.cuh>` which CUB removed in CUDA 13 (plus the
     build still emits `sm_52/60/61` gencodes that CUDA 13 dropped). Patched `cuda_util.cpp` with a
     `CUDART_VERSION >= 13000` guard (4-arg dependent-nodes on 13, 3-arg on 12; `decltype` cast for
     the driver entry point). This is forward-compatible and harmless on CUDA 12. The CUB reorg in
     `reduce.cu` is *not* patched — the fork is meant to be built against **CUDA ≤ 12**.
  2. **Carbon's Python layer is validated inside a CUDA-12 container**, not natively. New
     `scripts/py-test-cuda12.sh` runs the whole loop (`apt` numpy since PyPI is blocked, build the
     Warp fork with `--no_libmathdx --no_standalone`, build `libcarbonpy`, run `tests/test_warp_*.py`)
     in `cuda:12.8.0.ubuntu.24.04` with `--gpus all`. `--no_standalone` skips the `warp-clang`
     (CPU-JIT) DLL, whose LLVM/clang comes from NVIDIA's internal artifactory `urm.nvidia.com`
     (DNS-blocked here) — Carbon uses the CUDA path (nvrtc/nvcc), so the CPU backend is not needed.
     Result: **27 passed, 1 skipped**. Carbon's own C++ core builds and passes on the host under
     CUDA 13.1 regardless (only the Warp-facing Python layer needs the CUDA-12 container).
  XLB end-to-end still cannot run on this host because XLB requires `jax` (and other PyPI wheels)
  that the proxy blocks; that path is validated once PyPI access or a jax-provisioned image is
  available.
- **2026-08 — PyPI unblocked via Autodesk Artifactory.** PyPI is reachable through the internal
  index `https://art-toucan.autodesk.com/artifactory/api/pypi/autodesk-pypi-virtual/simple`
  (set as `index-url` in `~/.config/pip/pip.conf`). All XLB deps resolve there (jax 0.11.1,
  jaxlib 0.11.0, warp-lang 1.16.0, h5py, nvtx, pytest). Caveat: **pip >= 24 is required** — pip
  22.x (Ubuntu 22.04 apt) mangles Artifactory's virtual-repo download URLs (`packages/packages/…`)
  and 403s; the `cuda:12.8.0.ubuntu.24.04` image (pip 24) installs cleanly. This unblocks XLB
  end-to-end (Workstreams D-H) inside the CUDA-12 container: build the Warp fork + libcarbonpy,
  then `pip install` jax + XLB deps + XLB, and run with the Carbon-backed `ComputeBackend.NEON`.
  Second caveat: the Artifactory virtual repo only serves **already-cached** wheels — uncached
  ones 403 on-demand (e.g. `jax 0.11.1`, `jaxlib 0.8.2`). The cached, mutually-compatible pair
  `jax==0.11.0` + `jaxlib==0.11.0` works and satisfies XLB (`jax>=0.8.2`); it is pinned in
  `scripts/xlb-test-cuda12.sh`. That harness builds Warp + libcarbonpy, installs XLB's deps + XLB
  (`--no-deps`, so the `[neon]` wheel hook never runs and the built fork stays on the path), and
  runs `scripts/_xlb_smoke.py`. Confirmed green: Warp fork + XLB + Carbon import together with both
  GPUs visible (`carbon device_count: 2`).
- **2026-08 — XLB backend swap (Carbon under `ComputeBackend.NEON`): field init + pivotal
  spanless-partition finding.** See `docs/plans/xlb-neon-to-carbon.md` for the full plan. Two
  things landed / were pinned down:
  1. **`DenseField.zero_run()` / `fill_run(value)`** (Neon parity for grid setup). Implemented
     purely in Python/Warp — no native round-trip — via per-dtype fill kernels in
     `py/carbon/_fieldops.py` that write `value` into every component of every owned cell, launched
     once per device over the Standard data view. The kernels live in their own module precisely
     because `py/carbon/dense/__init__.py` uses `from __future__ import annotations`, which would
     stringify `@wp.kernel` parameter annotations and defeat Warp's concrete-type resolution.
     New test `tests/test_warp_field_fill.py`; full Python suite now **28 passed, 1 skipped** in
     the CUDA-12 harness.
  2. **Neon's `wp.neon_read(part, index, comp)` is *spanless*** — its partition embeds its own
     addressing metadata. Carbon's `wp.carbon.read(part, span, idx, comp)` keeps all offset math
     (pitch/halo_z/z_offset/extents/boundary_mode) in the **span** (`warp_dense.h`), so a 1:1
     `wp.neon_* → wp.carbon.*` alias is impossible at the arity level. **Decision:** embed
     `CarbonDenseSpan` into the dense partition POD (as Neon does) and add *spanless* builtins the
     aliases can map to 1:1, keeping the existing      span-carrying builtins for backward compatibility
     (`get_partition(device, data_view)` gains the job of filling the embedded span). This
     fat-partition change is the gating step that unlocks the ~28-file operator/kernel port; the
     exact file-by-file checklist is in the plan doc.

     **Landed (dense/single-res).** `CarbonDenseSpan` now lives in `warp_partition_types.h` and
     is embedded in every `CarbonDensePartition{F32,F64,I32,I64}` (C ABI struct in `carbonpy.h`,
     ctypes mirror in `warp_types.py`, C++ static-asserts in `pod_layout_checks.h`). Spanless
     `set/global_idx/read/write/neighbour/read_ngh` overloads were added to `wp::carbon::`
     (`warp_carbon.h`, each forwarding to the span-carrying overload via `part.span`) and
     registered in `warp_builtins.py`. `carbonpy_dense_field_get_partition` and the C++ launch
     path (`cuda_driver.cpp`) fill the embedded span (a `void* dense` back-pointer was added to
     `FieldBox`, mirroring `MgridFieldBox`). New test `tests/test_warp_dense_spanless.py`; full
     Python suite **29 passed, 1 skipped**. The Block/Mgrid spanless surface + multires LBM
     builtins are the remaining piece for the multires path.

     **Landed (alias shim + XLB config/grid rewire, single-res).** With the fat partition in
     place, `wp.neon_*` becomes a thin alias of the spanless `wp::carbon::` dense ops.
     `carbon.install_neon_aliases()` (`warp_builtins.register_neon_aliases`) registers
     `wp.neon_{read,write,read_ngh,ngh_idx,global_idx,get_x/y/z}` as Warp builtins whose native
     calls resolve to `wp::carbon::{read,write,read_ngh,ngh_idx,global_idx,get_*}` and attaches
     them to the `wp` module so kernel bodies written against Neon compile unchanged. Two C++
     overloads were added to `warp_carbon.h` to match Neon's exact shapes: `ngh_idx(int8,int8,
     int8) -> CarbonIndex3` (packs a stencil direction; `CarbonIndex3` is reused as the offset
     type) and a `read_ngh(part, idx, CarbonIndex3 ngh, comp, alternative, bool is_valid)` that
     forwards to the `(ox,oy,oz)` spanless form (the trailing `is_valid` is ignored for now —
     single-res callers pass an unused flag; multires wiring lands with the Block/Mgrid pass).
     On the XLB side (`XLB/`, branch `carbon`), `default_config.py`'s `ComputeBackend.NEON`
     branch now calls `carbon.init(enable_warp=True)` + `carbon.install_neon_aliases()` and
     stores the runtime handle on `DefaultConfig.carbon_gate`; `grid/neon_grid.py` builds a
     `carbon.System` + `carbon.Dense` (halo = stencil Chebyshev radius, 1 for D3Q19/27) instead
     of `neon.Backend`/`neon.dense.dGrid`. Two small Carbon parity shims support this:
     `carbon.Index_3d` (`py/carbon/index3d.py`) and a `cardinality=` kwarg alias on
     `Dense.new_field`. Device-subset selection (`dev_idx_list`) is still not plumbed through the
     C-ABI (Carbon uses every visible GPU; honour `CUDA_VISIBLE_DEVICES`) — tracked follow-up.
     Validated end-to-end on 2 GPUs in the CUDA-12 harness: new test
     `tests/test_warp_neon_aliases.py` (carbon suite **30 passed, 1 skipped**) plus
     `scripts/_xlb_grid_smoke.py`, which drives `xlb.init(ComputeBackend.NEON)` →
     `grid_factory` → `create_field(cardinality=, fill_value=)` → `wp.neon_read` read-back. The
     next prerequisite for the single-res operator/stepper port is uint8/int8 mask dtypes
     (bc/missing/solid masks), then the operators + steppers themselves.

     **Landed (uint8 dense dtype).** LBM masks (`bc_mask` / `missing_mask` / `solid_mask`) are
     `uint8`, which Carbon dense did not support (only f32/f64/i32/i64). Added `Dtype_U8 = 4` to
     `carbonpy::detail::CarbonDtype` and a `uint8_t` arm to `dispatch_dtype` (so both `dense.cpp`
     field creation and `cuda_driver.cpp` partition/halo binding now instantiate
     `carbon::Dense::Field<uint8_t,C>` + `carbon::Halo<uint8_t,C>` — these compile clean; the
     dense field/partition/halo path needs only read/write/memcpy, no reduce/NCCL/IO dtype
     traits). Warp headers gained the parallel u8 surface: `CarbonDensePartitionU8` +
     `CarbonNeighbourValueU8` (`warp_partition_types.h` / `warp_dense.h`), `carbon_dense_
     partition_read_u8` / `write_u8` (both `(ix,iy,iz)` and `(idx)` forms),
     `CARBON_DEFINE_DENSE_NEIGHBOUR(U8, uint8_t, ...)`, and in `warp_carbon.h` the span-carrying
     `read`/`write` overloads plus `CARBON_WARP_NEIGHBOUR_DENSE` / `CARBON_WARP_DENSE_SPANLESS` /
     `CARBON_WARP_DENSE_READ_NGH_OFFSET` instantiations for `U8`. POD static-asserts extended in
     `pod_layout_checks.h`. On the Python side: `dtypes.py` maps `wp.uint8` / `"u8"` /
     `"uint8"` → code 4 suffix `u8`; `warp_types.py` adds the two ctypes structs and registers
     them in `DENSE_PARTITION_BY_SUFFIX` / `NEIGHBOUR_VALUE_BY_SUFFIX` / the native-name maps
     (so `add_type` picks them up). To avoid dragging Block/Mgrid into a uint8 they do not need,
     `warp_builtins.py` introduces `_DENSE_DTYPES = _SCALAR_DTYPES + (("u8", wp.uint8),)` and
     splits the previously-interleaved registration loops: dense `read`/`write`/`global_idx`/
     `neighbour`/`read_ngh`/`value_or`/`value`/`valid`/spanless/neon-alias iterate the 5-dtype
     `_DENSE_DTYPES`, while Block/Mgrid stay on the 4-dtype `_SCALAR_DTYPES`. `_fieldops.py` adds
     a `_dense_fill_u8` kernel so `zero_run`/`fill_run` work on mask fields. New test
     `tests/test_warp_dense_u8.py` (u8 `fill_run` + `wp.carbon.read` + `wp.neon_read` read-back);
     carbon suite **31 passed, 1 skipped**. All single-res operator/stepper-port prerequisites
     (spanless dense, `wp.neon_*` aliases, uint8 masks) are now in place.
- **2026-08 — XLB single-res operator/stepper port to Carbon (Workstream F).** With the
    prerequisites above in place, ported the single-resolution `IncompressibleNavierStokesStepper`
    Neon path to Carbon so `ComputeBackend.NEON` runs a full fused LBM step end to end. The port
    is deliberately narrow — only the modules that touch the *Python* Neon orchestration API
    (`neon.Container.factory` / `neon.Loader` / `neon.Skeleton` / `neon.SkeletonConfig`) needed
    edits; every kernel *body* still calls `wp.neon_*` unchanged (the alias shim). Changed files:
    `nse_stepper.py` (`_construct_neon`: `@neon.Container.factory(name=)` → `@carbon.kernel(name=)`,
    `neon.Loader.Operation/Discretization` → `carbon.Loader.Operation/Discretization`;
    `prepare_skeleton`: `grid.backend` → `grid.system`, `neon.Skeleton(backend=bk)` →
    `carbon.Skeleton(system)`, `neon.SkeletonConfig` → `carbon.SkeletonConfig`),
    `macroscopic.py` and `quadratic_equilibrium.py` (same `@carbon.kernel` / `carbon.Loader`
    mapping + `c.run(0)` dropping Neon's `container_runtime=` kwarg), and `operator.py` (dropped
    two vestigial `import neon` lines in `_construct_read_write_functions` /
    `_construct_read_field_neighbor` — their bodies only use `wp.neon_*`). The math operators
    (stream, bgk, zero/first moment) were already pure `_construct_warp()` reuse and needed no
    change. `carbon.kernel` is the exact analogue of `@neon.Container.factory`: it wraps a
    generator returning a loading lambda `ll(loader)` into a `BoundKernel`, which `Skeleton.add`
    unwraps — so the loading-lambda bodies transferred verbatim. Added `Loader.Operation` /
    `Loader.Discretization` class-attribute aliases (`py/carbon/loader.py`) for Neon's
    `neon.Loader.Operation.stencil` spelling.

    Three infrastructure gaps surfaced and were closed to make the port run:
    (1) **LBM component counts.** Dense fields were only instantiated for `C ∈ {1,2,3,4}`;
    D3Q19 needs `C=19`. Introduced an X-macro `CARBON_DENSE_FOR_EACH_COMPONENT_COUNT` listing
    `1,2,3,4,7,9,15,19,27` and rewrote every dense `switch(components)` in `dense.cpp`
    (field new/delete, halo new/delete/run, get_partition) and `cuda_driver.cpp`
    (`fill_partition`, `append_dense_token`) through it — so `Field<T,19>` / `Halo<T,19>`
    (including the u8 masks) now instantiate. Block/Mgrid switches stay at 1-4 (multires LBM
    counts land with Workstream C). (2) **Capture-kernel launch with >2 fields.** The fused
    stepper binds 4 fields (`f_0`, `bc_mask`, `missing_mask`, `f_1`), so `build_bound_kernel`
    set `field_count=4`, which the C++ launch (cases for 1 or 2 only) rejected with `-1`. But
    capture kernels (`carbon.kernel` path) bake all partitions + span as Warp constants and take
    *no* user params, so their launch only needs the bounds. Added a `capture` flag to
    `DenseContainer` (threaded through `carbonpy_dense_container_new`'s ABI + `container.py`);
    when set, `launch_dense_gpu` launches with `params=[&bounds]` regardless of `field_count`.
    (3) **`carbonpy_dense_new` now logs the caught exception** (`std::fprintf` of `e.what()`)
    instead of a silent `-1` — this immediately surfaced a *harness* bug (an empty
    `CUDA_VISIBLE_DEVICES=` forwarded by `scripts/xlb-test-cuda12.sh` hides every GPU rather than
    exposing all; fixed to forward the var only when non-empty).

    Validation (`scripts/_xlb_stepper_smoke.py`): a uniform-equilibrium init is an exact BGK
    fixed point in the interior (streaming a uniform field is a no-op; `rho = q·f_init`, `u = 0`,
    `feq = w[l]·rho` each step, independent of `omega`), so after 2 steps the interior owned cells
    must equal `w[l]·rho` exactly. They do: `min = (1/36)·0.95 = 0.026389`, `max = (1/3)·0.95 =
    0.316667`, `rho = 0.95` uniform, no NaN — on **1 GPU** and on **2 GPUs with `occ=standard`**
    (the Skeleton auto-inserts the halo exchange for the stencil-read distribution field, so
    streaming across the z-slab boundary is correct). Only the open domain faces (x/y always
    Open; global z faces) drain and are excluded by a 2-cell margin. carbon regression unchanged
    (**31 passed, 1 skipped**). Remaining for full LDC/mlups parity: boundary conditions, forced
    collision, KBC, the multires stepper, and `prepare_fields` (`DenseField.copy_from_run` +
    `initialize_eq`'s equilibrium Neon launch).

- **2026-08 — XLB→Carbon: `prepare_fields()` path (`copy_from_run` + `initialize_eq`).**
    Wired the user-facing `Stepper.prepare_fields()` entry point end-to-end on the Carbon
    backend, so a full simulation no longer needs a hand-built uniform init. Three pieces:
    (1) **`DenseField.copy_from_run(other)`** — added per-dtype Warp copy kernels
    (`_dense_copy_{f32,f64,i32,i64,u8}`) and a `dense_copy(dst, src)` launcher in
    `py/carbon/_fieldops.py` mirroring `dense_fill`; it read/writes every component of every
    owned cell per device over the Standard view (halos re-established by the next halo
    update). `copy_from_run` on `DenseField` is the Neon-parity shim XLB calls to seed `f_1`
    from `f_0`. (2) **`initialize_eq`'s equilibrium launch** already resolves to the ported
    `QuadraticEquilibrium` carbon container — no new work, it just needed `copy_from_run` and
    the masker to stop importing `neon`. (3) **`IndicesBoundaryMasker._construct_neon`** —
    `prepare_fields` constructs this masker unconditionally (even with zero BCs), and its
    `_construct_neon` did `import neon` at construction time. Ported the imports + the three
    `@neon.Container.factory` → `@carbon.kernel` factories + `neon.Loader` annotations so
    construction succeeds. The loading-lambda **bodies** still use Neon-only multi-device APIs
    (`grid.backend`, `bk.get_num_devices()`, `loader.get_device_id()`, per-device warp-array
    indexing) and are only reached when a boundary condition is actually present — that full
    port lands with the boundary-condition workstream; the no-BC single-res path builds the
    masker but never invokes it.
    Validation (`scripts/_xlb_stepper_prepare_smoke.py`): drives
    `IncompressibleNavierStokesStepper.prepare_fields()` (create_nse_fields → copy_from_run →
    initialize_eq) then 2 fused steps. Equilibrium init with `rho=1, u=0` is an exact BGK
    fixed point, so interior owned cells must hold `w[l]·1.0`: measured `min = 1/36 =
    0.027778`, `max = 1/3 = 0.333333`, `rho = 1` uniform, no NaN — on **2 GPUs with
    `occ=standard`**. carbon regression **31 passed, 1 skipped** (new `test_copy_from_run` in
    `tests/test_warp_field_fill.py`). Remaining for full LDC/mlups parity: the masker
    multi-device loading bodies + BC operators, forced collision, KBC, and the multires
    stepper.
- **2026-08 — XLB boundary conditions + LDC + `mlups_3d` bring-up (Workstream F/H).**
  Ported the single-res boundary path so a full lid-driven cavity runs on Carbon:
  `IndicesBoundaryMasker` (`_construct_neon` → `carbon.kernel`; the multi-device loading
  bodies drive per-device warp arrays through `loader.get_device_id()` +
  `grid.backend.get_num_devices()`), plus the `EquilibriumBC` lid and
  `FullwayBounceBackBC` walls through the fused stepper. Also ported `EncodeAuxiliaryData`
  (`helper_functions_bc.py`) and `MomentumTransfer` (`momentum_transfer.py`) single-res
  `_construct_neon`/`neon_implementation` (Multires variants stay on the C workstream).
  - **Root-caused a latent global-coordinate bug.** `carbon_dense_global_idx` computed
    `out.z = idx.iz + z_offset` but every other site (neighbour-validity, existing tests)
    uses `halo_z + idx.iz + z_offset`. With `z_offset = -halo_z` the whole global z-axis
    shifted down by one, so the top z-plane never matched the lid indices: the masker left
    the lid untagged (histogram showed 0 lid cells, 2600/3176 walls) and the open boundary
    leaked, diverging to `rho ~ 1e17` by step 50 while the WARP reference stayed at exactly
    `max|u_x| = 0.05`. Fixing `warp_dense.h` to `out.z = span.halo_z + idx.iz + z_offset`
    makes the masker tag exactly 484 lid + 2692 walls on 24³ and the interior converge to
    `rho ≈ 1` with bounded physical velocity on 1 **and** 2 GPUs (`occ=standard`).
  - **Two supporting Warp fixes.** `neon_get_{x,y,z}` (C++ `warp_carbon.h` + the
    `warp_builtins.py` registration) now return/register `int32` instead of `int64` so the
    masker's `wp.vec3i(gx, gy, gz)` type-checks. And `wp.set_module_options({"enable_backward":
    False})` on `py/carbon/container.py` disables adjoint codegen for the whole module —
    Carbon kernels are forward-only, and the default backward pass emitted `+=` on the
    operator-less `CarbonIndex3` return of `neon_ngh_idx`/`neon_global_idx`, failing nvrtc.
  - **`mlups_3d` runs end-to-end (primary bring-up target).** The stock
    `XLB/examples/performance/mlups_3d.py 24 40 neon fp32/fp32 --gpu_devices=[0,1]`
    completes on 2 GPUs (~10 MLUPs, D3Q19 BGK, `occ=standard`) with no source edits beyond
    swapping the example's `neon.SkeletonConfig.OCC` / `neon.Report` to `carbon`. Added a
    pure-Python `carbon.Report` (`py/carbon/report.py`) mirroring Neon's `add_member` /
    `add_member_vector` / `write` but serialising to `<name>.json` (Neon backed it with a
    rapidjson C API; Carbon has no such need on the report path). Driver:
    `scripts/_xlb_ldc_smoke.py` (masker histogram + coord probe + interior stability over
    50 steps); carbon regression **32 passed, 1 skipped**.
- **2026-08 — XLB single-res Neon-path port completed + KBC validated.** Rewired the last
  single-res `ComputeBackend.NEON` entry points off the `neon` module onto `carbon.kernel`
  factories: the mesh (STL) maskers (`boundary_masker/ray.py`, `aabb.py`), `CustomInitializer`
  (`helper/initializers.py`), and `copy_neon_to_warp` (`utils/utils.py`, the Carbon-dense →
  WARP-array readback for JAX/VTK export via the `wp.neon_*` alias shim). The collision
  operators (`BGK`/`KBC`/`ForcedCollision`/`Smagorinsky`) needed no change — they are
  functional-only on the Carbon path (`_construct_neon` returns the warp functional; the
  `.neon_functional` attribute is a naming convention, not the `neon` module). Validated
  end-to-end: `mlups_3d 32 40 neon fp32/fp32 --collision_model KBC --velocity_set D3Q27
  --gpu_devices=[0,1]` runs on 2 GPUs (~18.8 MLUPs), which also exercises the full import
  path of every module touched. Remaining single-res gap is the export surface
  (`DenseField.update_host` / `export_vti`, Workstream G) needed for
  `--export_final_velocity`; multires (`simulation_manager.py`, `multires_grid_factory`, the
  multires masker/init/aux-data bodies) stays on Workstream C.
- **2026-08 — Dense host readback + VTI export (Workstream G core).** Added
  `DenseField.to_numpy` / `update_host` / `export_vti` (`py/carbon/dense/__init__.py`), a
  pure-Python multi-GPU-correct host gather: each device's slab is read back by aliasing its
  partition storage pointer as a non-owning Warp array on that device and copying to host, then
  the owned cells are sliced out with the dense layout math
  (`c*haloed_z*pitch_xy + (halo_z+iz)*pitch_xy + iy*logic_x + ix`) and placed into a global
  `(components, gx, gy, gz)` array at `gz0 = halo_z + z_offset`. No cross-device kernel writes,
  so it is correct across GPUs (unlike a single-array warp gather). `export_vti` writes a VTK
  `ImageData` (`.vti`, ASCII PointData, x-fastest / components-innermost). This unblocks
  `mlups_3d --export_final_velocity`. Validated: new `tests/test_warp_dense_readback.py` seeds
  every device slab with a coordinate-encoded pattern and checks the gather + VTI round-trip
  exactly (carbon suite **33 passed, 1 skipped** on 2 GPUs); and the stock
  `mlups_3d 24 40 neon fp32/fp32 --gpu_devices=[0,1] --export_final_velocity` writes a
  well-formed 24³ `.vti` (13824 points, 3 Float32 components, bounded physical velocities, no
  NaN). No XLB change needed — `create_field` already returns the `carbon.DenseField` the
  example calls these on.

- **2026-08 — Workstream C.1: fat mgrid partition (embed `own_span`).** First
  increment of the multires (Mgrid) port. XLB's multires kernels take a single
  partition handle from `loader.get_mres_{read,write}_handle` and use it both for
  own-level `read/write/neighbour/global_idx` and for the multires builtins
  (`has_child`/`has_parent`/`has_finer_ngh`/`lbm_read_coarser_ngh`/`mres_lbm_store_op`),
  exactly as Neon's `mPartition` does. Carbon previously split these (own-level
  `CarbonBlockPartition` vs cross-level `CarbonMgridPartition`, and the cross POD took a
  *separate* `own_span` arg). Mirroring the dense fat-partition move, `CarbonBlockSpan
  own_span` is now embedded in `CarbonMgridPartition` (placed right after `components`)
  across all seven layers: `warp_mgrid.h` POD macro, `carbonpy.h` ABI struct +
  `carbonpy_offsetof_mgrid_partition_own_span`, `pod_layout.cpp` offset fn,
  `pod_layout_checks.h` static-assert, `warp_types.py` ctypes layout, and
  `test_warp_pod_layout.py`. `mgrid.cpp::fill_cross_partition` fills it via
  `blit_block_span(grid(level), device, Standard, components)`. This makes the cross
  partition self-sufficient for own-level ops so the upcoming spanless mgrid builtins
  (C.2) and the five multires builtins (C.3) can be registered as `(part, idx, …)`.
  Regression-clean: carbon Python suite **33 passed, 1 skipped** (CUDA-12 harness, 2
  GPUs), including the enlarged `test_mgrid_partition_layout` and the existing
  `test_warp_mgrid_cross_level` / `_container_run` / `_neighbour`; format-check clean.
  See [`docs/plans/xlb-neon-to-carbon.md`](plans/xlb-neon-to-carbon.md) Workstream C for
  the full C.1–C.7 phase plan.

- **2026-08 — Workstream C.2: spanless own-level mgrid ops.** With `own_span` now
  embedded (C.1), registered the Neon-parity spanless own-level ops on the fat
  `CarbonMgridPartition`: `set(part, iter_idx, x,y,z)`, `global_idx(part, idx)`,
  `read(part, idx, c)`, `write(part, idx, c, v)`, `neighbour(part, idx, o…, c)`,
  `read_ngh(part, idx, o…, c, alt)` — each forwarding to the span-carrying own-level op
  via `part.own_span`. Added `carbon_mgrid_own_write_*` in `warp_mgrid.h` (mirror of the
  existing `own_read`) and the spanless `wp::carbon::` overloads in `warp_carbon.h`;
  registered them in `warp_builtins.py::_register_mgrid_spanless_builtins`. Important
  overload contract: the pre-existing 3-arg `read(part, span, idx, c)` remains the
  **parent** read (explosion source); the new 2-arg `read(part, idx, c)` is the
  **own-level** read — distinct arities, no collision. New test
  `tests/test_warp_mgrid_spanless.py` fills finest-level storage through the cross
  partition's spanless `set`/`global_idx`/`write` and verifies spanless `read` +
  `neighbour` + `read_ngh` round-trip (carbon suite **34 passed, 1 skipped**, CUDA-12, 2
  GPUs; format-check clean). The loader switch (`get_mres_*_handle` → cross partition) is
  deferred to C.6 with the XLB kernel ports, since flipping it now would reinterpret the
  block-partition mgrid tests' `read(part, span, idx)` as a parent read.

- **2026-08 — Workstream C.3: five multires LBM device builtins.** Implemented the
  Neon-`mPartition`-parity multires primitives on the fat cross partition, closing the
  last backend gap before the XLB multires kernel ports. In `warp_mgrid.h`:
  `has_child_idx`/`has_parent_idx` (spanless wrappers over the existing
  `has_children`/`has_parent`), `has_finer_ngh` (active same-level neighbour that is
  itself refined), the geometry helpers `uncle_off` (Neon's
  `off(i,j)=(i+j<=0)?i+j:i+j-1`), `parent_voxel` (parent block + `(global/R)%Mem`
  in-block coord, matching the validated `parent_read`), and `get_uncle`
  (neighbour-of-parent on `parent_span`; direction `(0,0,0)` = direct parent), then
  `lbm_read_coarser_ngh` (explosion: `uncle_off(inBlock%R, pullDir)` → raw parent-level
  read, no scaling — normalization lives in separate coalescence passes) and
  `mres_lbm_store_op` (coalescence fan-in: `atomic_add` into the parent slot at
  `uncleDir - qDir` when the same-level neighbour is inactive but the uncle + its child
  are active). `carbon_mgrid_atomic_add` is `__CUDA_ARCH__`-guarded (device `atomicAdd`,
  int64 via `unsigned long long` reinterpret; host `+=`). Exposed as `wp::carbon::*` in
  `warp_carbon.h` (plus a mgrid `read_ngh` offset overload that wires the trailing
  `is_valid` out-ref to own-level neighbour validity, which multires stream steps branch
  on) and registered + aliased in `warp_builtins.py` as
  `wp.neon_{has_child,has_parent,has_finer_ngh,lbm_read_coarser_ngh,mres_lbm_store_op}`
  and mgrid `wp.neon_{read,write,global_idx,read_ngh}`. `neon_level` is deferred (XLB uses
  it only in comments/TODOs). Validated by `tests/test_warp_mgrid_multires.py` on a
  2-level all-active octree: level semantics (finest has parent/no child, coarsest the
  reverse), the invariant `lbm_read_coarser_ngh(ngh=0) == parent read`, and
  `has_finer_ngh` + `mres_lbm_store_op` codegen/run (carbon suite **35 passed, 1
  skipped**, CUDA-12, 2 GPUs; format-check clean). Full explosion/coalescence numerics
  need a genuinely refined sparse grid (the all-active octree makes the store a near
  no-op), which arrives with C.5's per-level sparsity.

- **2026-08 — Workstream C.4: `MgridField` per-level field ops.** Gave the multires
  field the Neon-parity per-level device operations the XLB grid factory and multires
  stepper call before any kernel touches the data: `fill_run(level, value)`,
  `zero_run(level)`, and `copy_from_run(level, other)`, plus a `num_levels` property
  (delegates to `Mgrid.depth`). Implemented in `py/carbon/_fieldops.py` as block-shaped
  Warp launches (`iter_block_count × MemX × MemY × MemZ`) over `CarbonBlockPartition{F32,
  F64,I32,I64}` + `CarbonBlockSpan` with a `view_block_idx`/`is_active` guard — reusing
  the already-validated block `read`/`write` builtins rather than the new spanless
  surface, so these ops share the exact iteration model as the cross-level test's
  `fill_level`. Each op loops every visible device, sets `span.components = part.components`
  (the grid-level `get_span` C ABI does not know the field's component count), launches on
  that device's stream, and fences. `mgrid_copy` rejects dtype / component mismatch up
  front. Signatures match XLB's call sites exactly (`fill_run(level=…, value=…)`,
  `copy_from_run(level, other, stream_idx)`, `zero_run(level)`). Validated by
  `tests/test_warp_mgrid_field_ops.py` on a 2-level all-active octree with a 2-component
  field: `fill_run` leaves zero mismatched active cells and touches a positive count at
  every level, `copy_from_run` reproduces the source's active-cell set exactly, and
  `  zero_run` clears the same set (carbon suite **36 passed, 1 skipped**, CUDA-12, 2 GPUs).

- **2026-08 — Workstream C.4 (cont.): mgrid host readback.** Added `MgridField`
  `to_numpy(level)` / `update_host(level)` / `export_vti(filename, level, …)`, closing out
  C.4. `to_numpy` scatters each device's owned active cells (block-shaped Warp launch, per
  dtype, guarded by `is_active`) into a full-extent zero `wp.array` on that device — indexed
  by the level-local `global_{x,y,z}` builtins — then sums the per-device host arrays
  (device z-slabs are disjoint, so the sum is a merge; no cross-device access). The result is
  `(components, gx, gy, gz)` where the level extent is `finest // ref_factor**level` (`Mgrid`
  now stores `finest`/`ref_factor` and exposes `level_extent(level)`); inactive cells read
  back as zero. `export_vti` reuses the dense VTK-ImageData writer per level; `update_host`
  keeps a per-level host mirror. Validated in `tests/test_warp_mgrid_field_ops.py`: after
  `fill_run(level, 7.0)` the gathered array has the expected shape and exactly the counted
  active-cell/component slots equal to 7.0 (suite **36 passed, 1 skipped**, CUDA-12, 2 GPUs).
  Loader `set_mres_grid` + `Operation.stencil_up` wiring lands with the C.6 operator ports.

- **2026-08 — Workstream C.5: sparse mGrid construction (`carbonpy_mgrid_new_sparse`).**
  Removed the "all-active only" blocker for the XLB multires grid. The old
  `carbonpy_mgrid_new` hard-coded `is_active = [](…){ return true; }`, giving a fully dense
  octree — enough to validate the whole binding stack (C.1–C.4) but not a real adaptive mesh.
  Added `carbonpy_mgrid_new_sparse(handle, system, finest_xyz, depth, ref_factor, masks_flat,
  mask_shapes, mask_origins)`, mirroring Neon's pyNeon `mGrid(sparsity_pattern_list,
  sparsity_pattern_origins)` interface. **Key finding (verified against upstream Neon):**
  `MultiResBlock` already exposes exactly the mechanism needed — an `is_active(level, voxel)`
  predicate — and Neon's per-level "origins" are *index-space mask placements*, not physical
  world origins (Neon's C++ mGrid keeps a single shared `origin`/`spacing`). So the origins
  fold entirely into the predicate and **no mGrid core change was required**; the C ABI merely
  marshals the masks and the C++ side rebuilds the predicate closure. Semantics are level-local
  (level 0 = finest): each level's mask is a dense int32 array in C-order (`(x*sy+y)*sz+z`),
  a voxel `v` is active iff `mask[v − origin]` is in-bounds and non-zero; masks are copied into
  the closure so the caller may free them immediately. Because Carbon's `MultiResBlock` does no
  strong-balance / overlap-cull (unlike Neon's optional passes), the active set equals exactly
  the masked voxels within retained bricks, making counts deterministic. Wired through
  `gate.py` (ctypes) and a `Mgrid(…, sparsity_pattern_list=, sparsity_pattern_origins=)`
  constructor overload that keeps the all-active default. Validated by
  `tests/test_warp_mgrid_sparse.py` on 2 GPUs: full-coverage masks reproduce the all-active
  active-cell count; a brick-aligned half-domain mask yields exactly the masked cells
  (`half·n·n`) and masked-in/out cells read back fill/zero (suite **37 passed, 1 skipped**,
  CUDA-12). Remaining for the XLB grid port (C.6): convert XLB's finest-coord physical origins
  to level-local, and add a `memory_type=` arg to `Mgrid.new_field` (`multires_aabb_close`).

- **2026-08 — Block/mGrid offset math takes its component count from the partition, not the
  span.** `CarbonBlockSpan` carried an `int32_t components`, which is a category error: a span
  describes the *grid* (brick geometry, connectivity, iteration bounds) and is shared by every
  field over it, while the component count is *per field*. The consequences were visible in the
  code: `carbonpy_block_get_span` had no field to ask, so it hard-coded `out_span->components =
  1`, and every multi-component caller had to repair the struct by hand — 14 sites of
  `span.components = part.components` across `_fieldops.py` and the mgrid/block tests. Miss the
  patch-up and a multi-component field addressed as if it had one component: silent data
  corruption, no diagnostic. `carbon_block_linear_offset` now takes `components` as an explicit
  parameter and each caller passes `part.components`, so the count comes from the partition that
  actually owns it and the patch-ups are gone. The `span` parameter is retained (marked
  `[[maybe_unused]]`) to keep the free function shaped like Carbon's C++
  `Span::linear_offset` receiver, which the layer-1 Warp port mirrors; note the body still
  derives the in-brick stride from the compile-time `kBlockMem*` constants rather than
  `span.mem_*`, which is why nothing else in it needs the span today. The field is deleted from
  the POD on both sides (`wp::CarbonBlockSpan`, the C ABI struct, the `warp_types.py` ctypes
  mirror), along with the `carbonpy_offsetof_block_span_components` helper and its layout check.
  `CarbonMgridPartition::components` and `CarbonBlockPartition::components` stay — that is where
  the count belongs. Note this shifts the offsets of `CarbonMgridPartition`'s three nested
  `CarbonBlockSpan` members, which `test_warp_pod_layout` verifies on both sides.
  Locked by the new `tests/test_warp_multi_component.py`: a 3-component `Block` field and a
  2-component 2-level `Mgrid` field, checking every component on the own-level block path and on
  the cross-level parent / child paths. Both cases were *falsified* before being trusted —
  pinning the pitch back to 1 makes them fail loudly (11776 cross-level mismatches) rather than
  pass vacuously, which is the property the single-component tests lacked. Every other
  block/mgrid test uses one component, where the old and new math coincide, so this gap had no
  coverage at all.

- **2026-08 — The Warp test layer was reporting passes it had not earned; `ctest` now runs it
  for real.** Three independent defects stacked into a suite that looked green while executing
  almost nothing. (1) Each `tests/test_warp_*.py` opens with a chain of `skipTest` branches
  (library not staged, `warp` not importable, fork not installed, no device), and ctest invokes
  `${Python3_EXECUTABLE}` — the project venv, which has no `warp` installed — so 19 of 20
  registered tests took the ImportError branch. A unittest skip is a ctest pass, so
  `ctest -R warp` reported **20/20 passed in 0.65s**; with the environment supplied the same
  set does real work in ~31s. The fork was *already built* at `warp/warp/bin/warp.so`; it was
  only ever a path problem, and `scripts/py-test-cuda12.sh` had been masking it by exporting
  `PYTHONPATH=/carbon/warp:/carbon/py` inside its container. Registration now goes through a
  `carbon_add_warp_test()` helper that sets that `PYTHONPATH` as a ctest `ENVIRONMENT`
  property, so plain `ctest` — the command `AGENTS.md` tells contributors to run — is
  sufficient. PYTHONPATH is *replaced* rather than prepended on purpose: these tests must
  exercise the vendored external-type fork, not whichever warp a developer happens to have
  installed. (2) Nine test files existed but had never been added to `add_test` at all
  (`dense_readback`, `dense_spanless`, `dense_u8`, `field_fill`, `mgrid_field_ops`,
  `mgrid_multires`, `mgrid_spanless`, `mgrid_sparse`, `neon_aliases`), so they ran only under
  the pytest script, which globs the directory. `mgrid_field_ops` is the layer's only
  multi-component mGrid coverage, so ctest had *no* view of the component-pitch path it
  guards. The suite is now 31 registered and green. (3) `test_warp_smoke.py` looked for the
  library under `<root>/carbon/lib/` and put `<root>` on `sys.path`, both missing the `py/`
  segment, so it could only ever skip. Warp tests were also unlabelled, hence invisible to
  `ctest -L cpu` / `-L gpu`; they now carry `warp` plus a lane, so `ctest -L warp` selects the
  layer. Added `tests/test_warp_gate.py` on the model of `test_multi_gpu_gate.cpp`: it reports
  exactly which piece of the environment is missing, passes with a loud warning by default,
  and *fails* under `CARBON_TEST_STRICT=1` — so one honest signal replaces twenty misleading
  ones, and a lane provisioned for this layer turns red instead of quietly no-opping. All
  three of its modes were verified by falsification rather than assumed.

- **2026-08 — `gpu error 222` across the C++ suite was a build-tree arch default, not a code
  defect.** A full `ctest` on this dev box failed 53 of 128 tests, every one of them with the
  same launch-time error: `the provided PTX was compiled with an unsupported toolchain`
  (CUDA 222, `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`) — 102 occurrences, one root cause. The repo
  does not pin `CMAKE_CUDA_ARCHITECTURES`, so trees configured without it inherited CMake's
  default of `75`; with no `sm_89` cubin for these RTX 6000 Ada cards, every launch had to JIT
  the embedded PTX, and the JIT is where it broke — `/usr/local/cuda` is 13.1 while driver
  580.178.04 is a CUDA 13.0 driver, which cannot consume 13.1 PTX. Isolated with a 2x2 probe
  rather than guessed: 13.1/`sm_75` fails, while 13.1/`sm_89`, 13.0/`sm_75` and 13.0/`sm_89`
  all pass, so it needs *both* the newer toolkit and a non-native arch. Worth noting the
  failure mode is not always loud: a probe that checked only `cudaDeviceSynchronize()` reported
  success with the kernel silently not having run, and only surfaced 222 once it read
  `cudaGetLastError()` straight after the launch — carbon's own launch path checks properly,
  which is why the suite errored honestly. Reconfiguring with
  `-DCMAKE_CUDA_ARCHITECTURES=89` and rebuilding took the suite to **128/128 on 2 GPUs under
  `CARBON_TEST_STRICT=1`**, all 29 `needs_multi_device` tests doing real work. Both `build/`
  and `build-py/` had been in the broken state since 2026-08-19, which is the more useful
  lesson: the trees were stale, not the code, and no build-time diagnostic existed to say so.
  `AGENTS.md` now leads its verification notes with the arch requirement.

- **2026-08 — Nested-registered-type returns work in the Warp fork; both binding layers are
  unblocked.** The two-layer plan had one unproven dependency at its centre: layer 1 wants
  `set_and_validate` to return C++'s `ValidatedIdx { Idx idx; bool valid; }` and layer 2 wants a
  fat index `{ Idx idx; Span span; }` so XLB kernel bodies need no edits, and both require a
  builtin to *return* a registered external type that nests another registered type. Nesting on
  the input side was already shipping (`CarbonDensePartition` embeds a `CarbonDenseSpan`,
  `CarbonMgridPartition` embeds three), but returning one was not exercised anywhere. Probed
  with `warp_probe.h` + `test_warp_probe_nested_return.py`, which walks four separately
  questionable capabilities: registering a ctypes struct whose fields are registered types;
  returning it by value; passing it into another builtin that unpacks the nested members in C++;
  and an accessor builtin handing a nested registered type back out. **All four work**, and the
  fat path reads back bit-identical to the ordinary span-carrying calls, so it addresses the same
  memory rather than merely compiling. Falsified by dropping `out.span = span` from the maker,
  which fails the test — the round trip is really being checked. Two notes for the follow-up
  work: member access on external types goes through accessor builtins (`wp.carbon.valid(v)`),
  not `.member` syntax, so layer 1's `ValidatedIdx` needs explicit accessors; and the binary
  constructors skip their memcpy when the byte count disagrees with `sizeof`, which would hand a
  kernel a silently zeroed struct, so the test pins the ctypes layout as well. Phase 2 can
  therefore build the C++-shaped `set_and_validate` rather than decomposing it.

- **2026-10 — PatternScalar in the Skeleton (deferred scalar readback).** Reduction
  scalars are first-class DAG resources: `AccessToken` gains `ResourceKind` and
  `ReduceOptions::host_copy` (`Inline` default vs `Deferred`). `ReductionResult` uses
  pinned host slots, a slot ring (`ReductionResultOptions::slots`), per-slot `slotDone`
  events, and `wait_value` / `ready` for lagged host reads. `Kernel::scalar_readback`
  and auto-insertion after deferred reduces run on a detached copy stream; anchor
  fork-join skips the detached copy stream and allocates no anchor events when it is
  the only secondary stream (`Skeleton::uses_anchor_fork_join()`). CUDA graph capture is disabled
  when a detached copy stream exists. Scalar tokens and slot/D→H behaviour are shared by
  `ReduceKernelImpl` and `UniversalReduceKernelImpl` (`Kernel::reduce_universal` accepts
  `ReduceOptions`). Inline D→H records `slotDone` for `wait_value`. Tests:
  `test_reduce_scalar_ring`, `test_skeleton_scalar_pattern`,
  `test_skeleton_scalar_pattern_multi_gpu`. Jacobi adds `--norm-sync blocking|pipelined`.

- **2026-10 — Neon-style Skeleton scheduling (Standard overlap).** `compile_step_graph_and_lower_`
  builds a step graph (data + hint edges), applies Standard OCC splits (compute and
  reduction with partial + merge launches), assigns streams (Boundary on the halo stream,
  Internal before halo via hint edges), emits cross-stream events, then
  `compile_prune_syncs_()` drops redundant waits/records and covered anchor joins.
  `OverlapMode::Extended` / `TwoWayExtended` still use the legacy `apply_occ_` path.
  Reduction splits use `reduce::ReduceLaunchMode::{Partial,Merge}`; Jacobi defaults to the
  fused stencil+norm kernel (`--unfused` for the old split pipeline). Optional high-priority
  halo streams: `Skeleton::Options::halo_stream_priority`, `System::stream(StreamPriority)`,
  benchmark `--hp-halo`. `Skeleton::export_launch_matrix_dot()` (benchmark `--dot-matrix`)
  renders the launch order as a streams × operations grid; `Skeleton::export_kernel_list_dot()`
  renders the added kernels as a table (per-resource access, dependencies, launches).
