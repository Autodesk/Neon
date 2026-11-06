# Carbon review: the distributed backend (PR #76) and its surroundings

Reviewed on 2026-11-06. Code base: branch `hp-halo-priority` (`bc134e9`), which is
`Autodesk/Neon` `carbon` `d4e3d06` (#76) plus the halo-priority fix.

The profiling notes and traces cited below are not in this repo. They are in the notes repo
`massimim/ai-repos`, branch `carbon-distributed-skeleton`: `overlap-halo-priority.md`,
`brick-compute.md` and `hp-halo-ab/`.

## Method

The review was split into five areas, each read end to end by a read-only reviewer:
1. the DenseDistributed core (layout, span, process grid);
2. the halo exchange and communication (`HaloDistributed`, pack plans, NCCL, `Distributed`);
3. kernel launch and reductions;
4. distributed I/O and checkpoint;
5. tests, benchmarks, Docker and docs.

I then re-read the code for the high-impact claims. Items marked **verified** were confirmed in
the source by me. Performance causes are hypotheses unless they come from our own traces
(`overlap-halo-priority.md`, `brick-compute.md`). Nothing here was built or run for the review
itself.

Paths are relative to the top of this repo.

## TL;DR: the ten things most worth fixing

| # | issue | kind | effort |
| --- | --- | --- | --- |
| 1 | Chunk-size validation runs only on the local chunk, so one rank can throw while the others hang (§1.1) | bug (hang) | small |
| 2 | Boundary view visits cells twice on pencil/brick with a z-only stencil and thin chunks (§1.2) | bug (wrong results) | small |
| 3 | `dense_distributed_dist`, the only GPU test of pencil/brick, has a wrong expectation, stops at its first failure, and silently skips unless world is 4 or 8 (§4.1) | test | small |
| 4 | Errors are never agreed across ranks: NCCL group not closed on throw, no async-error check, I/O failure on one rank only (§2) | robustness | medium |
| 5 | Docker images install an unpinned NCCL (got 2.30 for CUDA 13.2, needs driver ≥ 580) and the CUDA configs set no GPU architecture (§4.4) | build | small |
| 6 | VTK shards leave a one-cell gap at every rank boundary in ParaView; physical origin/spacing dropped (§5.1) | bug (output) | small–medium |
| 7 | HIP on AMD: `BlockAtomic` sums of `double` / `int64_t` compile to nothing (§1.3) | bug (wrong results) | small |
| 8 | Halo passes always run X → Y → Z in sequence, even for stencils with no diagonal neighbours; pack kernels are uncoalesced on X faces (§3.2, §3.3) | performance | medium |
| 9 | Brick: rows are not aligned (pitch = owned + 2) and Boundary x-faces are uncoalesced (§3.1) | performance | medium |
| 10 | The distributed LBM and CG benchmarks validate nothing, and LBM timing is rank 0 only, with no barrier (§4.3) | benchmark | small |

Also worth knowing, from earlier work: the Skeleton never gave the halo a high-priority stream
(fixed on this branch; +20% MLUPS on pencil-4 N=256, see `overlap-halo-priority.md` §5.2).

---

## 1. Correctness bugs

### 1.1 Validation that throws on only some ranks (verified)

- **Where:** `src/dense_distributed.cpp:196-225`; `Dense` has the same pattern at
  `src/dense.cpp:192-203`.
- **What:** both thickness checks loop over `spans_`, which holds only this rank's chunk under
  MPI:
  - `lz < 2*halo.z` for the legacy slab;
  - `owned < halo` for multi-axis layouts.
- **Why only some ranks throw:** `split_extent` gives the remainder to the first coordinates,
  so the last rank on an axis has the thinnest chunk. Example: z = 7 over 4 ranks gives
  2, 2, 2, 1, so only rank 3 throws and the others hang at the next collective.
- **Contrast:** the owned-count check just above (`:173-180`) already loops over every global
  chunk.
- **Fix:** move both thickness checks into that global loop. They are deterministic and cheap,
  so every rank then throws together.

### 1.2 Boundary view double-visits cells (verified)

- **Where:** `include/carbon/detail/dense_distributed/span.h:188-190` and `:491-494`.
- **What:** with `halo.x == halo.y == 0` the Boundary view uses the legacy two-band z path:
  `bz = iz < halo.z ? halo.z + iz : owned_z - halo.z + iz`. Its 2×halo guard only runs when
  `legacy_slab` is true (`src/dense_distributed.cpp:198`). Non-legacy layouts only require
  `owned.z >= halo.z` (`:217`).
- **Trigger:** a pencil or brick, or a periodic-X/Y slab, with a z-only stencil and
  `owned.z = 1`. Then `iz=0` and `iz=1` both map to `bz=1`.
- **Effect:** any non-idempotent Boundary kernel (`a += …`, reductions) gives wrong results.
  The host test only skips this case for the legacy slab, and no case has zero x/y radius.
- **Fix:** take the two-band path only for `legacy_slab`, or clamp the bands as
  `shell_bands_()` does. Add the case to `test_dense_distributed_host.cpp`.

### 1.3 HIP: atomic `double` / `int64_t` sums do nothing (verified)

- **Where:** `include/carbon/detail/backend/cudalike/dense_distributed_reduce_kernel.h:59,68`.
  The same copy-pasted guard is in `dense_reduce_kernel.h`, `dense_disg_reduce_kernel.h` and
  `block_reduce_kernel.h`.
- **What:** the guard is `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600`, and the `#else`
  branch discards the value. HIP device compilation does not define `__CUDA_ARCH__`. The repo's
  own `cudalike/atomic.h` already adds `|| defined(__HIP_DEVICE_COMPILE__)` for this reason.
- **Fix:** call `atomic.h` from here, and make the silent `else` branches a dependent
  `static_assert`.

### 1.4 The opt-in 2-D stage-1 reduce ignores y (verified)

- **Where:** `cudalike/dense_distributed_reduce_kernel.h:190-191,229-230` call
  `set_and_validate(view, ix, 0, iz)`; the grid is `gx*gz`.
- **What:** this is only correct for views with y extent 1. On a multi-axis Boundary
  (extent `(128, rows, 1)`) it reduces only the first 128 shell cells. On Standard/Internal
  views with y > 1 it reduces only the y=0 plane. Nothing rejects it.
- **Related:** `reduce.h:277-283` only lets 32×32 through, so the 8×8, 16×16 and 64×64
  dispatch cases are dead (64×64 would also be an invalid launch).
- **Fix:** reject `iter_extent(v).y != 1` in the `ReduceKernelImpl` constructor, or use a real
  3-D mapping. Remove the dead cases.

### 1.5 SYCL: split-reduction merge throws (high confidence)

- **Where:** `include/carbon/detail/reduce_kernel_impl.h:568-611`. `launch_merge_on_device` is
  CUDA/HIP only and throws `logic_error` elsewhere.
- **Effect:** stencil reductions under any overlap mode (Partial + Partial + Merge) fail on
  SYCL. `skeleton_reduce_split` is registered for SYCL too.
- **Also:** `src/dist_reduce.cpp:213-223` calls `q.wait_and_throw()` after every cross-rank
  reduction, and its lambdas capture by reference, so removing the wait alone would leave
  dangling references.
- **Fix:** implement Merge with a `single_task`, and capture by value so the wait can go.

### 1.6 Minor correctness items

- **Missing int32 stride guard:** `Partition::cursor` narrows strides to `int32_t`
  (`detail/dense_distributed/partition.h:270-280`). `Dense` rejects layouts that overflow
  this; `DenseDistributed` doesn't, and haloed storage size is never overflow-checked.
- **Min/Max identities:** they are `numeric_limits::max()` / `lowest()` (`reduce.h:135,157`),
  so a Min over all-`+inf` values returns `FLT_MAX`. NaN survival depends on argument order.
  For floating types use `±infinity()` and document the NaN policy.
- **Run-based X/Y faces span the full haloed range:**
  - Where: `detail/halo_distributed_build.h:129-153,189-213`.
  - What: they overwrite exterior halos on open axes with the neighbour's exterior cells.
    The packed path avoids this.
  - Why the test misses it: every rank uses the same sentinel, so sentinel-over-sentinel
    is invisible. Use `kSentinel - rank`.

## 2. Robustness: failures that become hangs

- **No collective agreement on errors.** Several failure paths throw on one rank only, and the
  peers then block in the next NCCL call or MPI barrier:
  - the validation in §1.1;
  - VTK write failures (§5.3);
  - the per-rank test checks (§4.1);
  - the benchmarks' `return EXIT_FAILURE` on the failing rank only.

  A small helper would fix all of these: `all_ok(bool)`, which does an `MPI_Allreduce` (logical
  AND) and throws the same error on every rank. Call it at construction, after I/O, and in
  tests. `MPI_Abort` is the blunt fallback.
- **NCCL group left open on throw (verified).** `src/dist_transfer.cpp:241-256` calls
  `check_nccl(ncclSend/Recv)` inside `ncclGroupStart … ncclGroupEnd`. A throw skips
  `ncclGroupEnd`, so later NCCL calls are silently deferred. Use an RAII group guard.
- **No asynchronous error handling (verified: nothing calls `ncclCommGetAsyncError`,
  `ncclCommFinalize` or `ncclCommAbort`).** A dead peer hangs in `stream.wait()`. Add an
  optional watchdog that polls the async error and aborts the communicator, and use
  `Finalize` + `Destroy` on shutdown.
- **Device pinning (`src/distributed.cpp:166-176`, verified):**
  - fmt prints the literal `%%`;
  - the warning fires on the normal case of several ranks on an 8-GPU node;
  - `local_size > visible` (ranks sharing a GPU) only warns, then fails later in
    `ncclCommInitRank`.

  Log at info level when `visible >= local_size`, and throw a clear error when it is smaller.
- **`HaloDistributed` is not safe across streams.** Copies share `stage_` and the fence events
  (`halo_distributed.h:435,489,495`). Ping-pong fields are fine because each has its own
  instance. Document "one stream at a time per instance".

## 3. Performance

Ordered by expected payoff on the LBM workload.

### 3.1 Brick layout: misaligned rows and uncoalesced Boundary x-faces

- **Misaligned rows (verified layout):**
  - Row pitch is `haloed.x = owned + 2·halo.x` with no padding
    (`detail/dense_distributed/layout.h:125-126`), and owned/Internal cells start at
    `halo.x` / `2·halo.x`.
  - For fp32 at 256³, rows are 1032 B and the first owned cell is at +4 B, so warps touch
    5 sectors instead of 4.
  - Component planes are misaligned too: 258³·4 B ≡ 32 (mod 128).
  - Pencil and slab are unaffected (`halo.x = 0`). This is consistent with brick Internal
    costing 155 vs 127 ps/cell (`brick-compute.md`).
  - **Fix:**
    - add a `y_stride()` to `Span`, with pitch = `round_up(owned.x + 2·halo.x, 32)`;
    - give each component plane a lead pad, so buffer x = `halo.x` is aligned;
    - round component strides to 256 B.

    That costs about 6% memory at 512³. The hard-coded `haloed.x` stride appears in
    `span.h`, `partition.h`, `field.h`, `halo_distributed_build.h`, `halo_pack_plan.h` and
    `io_vtk_distributed.inl`. Make it opt-in and measure.
- **Boundary x-faces (verified mapping):**
  - `shell_cell_` (`span.h:427-443`) maps x-faces one cell per row, so the 32 lanes of a warp
    are 32 rows apart. About a third of the brick shell is x-face cells.
  - This explains part of the 359 vs 37 µs Boundary gap; the estimate covers 3–5× of the 10×.
  - **Fix:** launch x-faces separately (y along threads), and start them first or on the halo
    stream. For 8 GPUs, prefer a 1×2×4 or 1×4×2 pencil over a 2×2×2 brick: the shell size is
    similar and every face is row-contiguous.

### 3.2 Halo passes always run X → Y → Z in sequence

- **Where:** `halo_distributed.h:255-296`; `kernel_dense_distributed.inl:80` builds the halo
  without the stencil, so the halo never knows whether diagonal neighbours exist.
- **What:** for face-only stencils (7-point, or any stencil without edge/corner directions)
  the faces are independent.
- **Fix:** pass the stencil to the halo factory. When no stencil point has two or more
  non-zero axes, use one fused gather, **one** NCCL group for all axes, and one fused scatter.
  For D3Q19 the edges are still needed, but the corner columns can be dropped.
- **Expected saving:** about 13–15 µs of the ~130 µs pencil-4 exchange (two of three
  gather→NCCL gaps, two scatter→gather gaps), plus host enqueue time. With `--hp-halo` the
  halo is already hidden at N=256, so this matters most where it is not (N=512, two nodes).

### 3.3 Pack/unpack kernels

- **Where:** `detail/halo_pack_plan.h:123-132,350-388,476-510`.
- **What:**
  - one launch per side, per self-wrap and per block of up to 32 components, for gather and
    again for scatter;
  - X faces: consecutive threads are `haloed.x` apart, so 32 sectors per warp;
  - Y faces: `grid_x = 2` for 258-wide rows, so about half the blocks are almost empty.
- **Fix:**
  - one launch per pass, with an array of `PackKernelArgs` indexed by `blockIdx.z`;
  - warps along y for X faces, or a shared-memory transpose;
  - several z-planes per block for thin rows.

### 3.4 Run-based fallback: one copy or NCCL message per row

- **Where:** `halo_distributed.h:869-876,974-977,1010-1058`; `transfer.h:278-297`.
- **What:** X faces become `haloed.y·haloed.z` runs, about 66k `cudaMemcpyAsync` or
  `ncclSend` calls per face at N=256. This is the only path on SYCL, for single-process
  multi-GPU, for odd `sizeof(T)` and with `CARBON_HALO_PACK=0`.
- **Fix:**
  - one `cudaMemcpy3DAsync` per face for intra-rank copies;
  - port the gather/scatter kernels to SYCL (the index math is already backend-neutral);
  - at least warn once when falling back.

### 3.5 Host overhead and CUDA graphs

- **Per-call allocations:** `dist_transfer.inl:61-68` allocates two `std::vector`s and
  converts to bytes on every call. Precompute these at construction.
- **No-op fences:** on one GPU per rank, the per-pass fences (`halo_distributed.h:251-254,
  276-280`) are a no-op `record` + `wait` pair. Skip them when `gpu_group().size() == 1`.
- **CUDA graph capture:** the exchange is close to capturable (buffers are preallocated, and
  NCCL point-to-point can be captured). The blockers are the opt-in `pass_barrier_` and the
  MPI transport's `stream.wait()`. Capturing the whole Skeleton iteration (Internal + halo +
  Boundary) would remove most of the 5–6 µs gather→NCCL host gaps.

### 3.6 Halo and reductions share one NCCL communicator (hypothesis)

- **Where:** `src/system.cpp:30-34` creates one communicator per process.
- **What:** if NCCL runs operations on one communicator in issue order across streams, a
  high-priority halo cannot overtake a pending allreduce (CG, Jacobi norm).
- **Fix:** check with nsys on CG, then consider a halo communicator via `ncclCommSplit`.

### 3.7 Reductions

- **Per-thread decode cost:** stage 1 decodes the flat index with four 64-bit div/mod per
  thread (`cudalike/dense_distributed_reduce_kernel.h:104-109,145-150`). The Boundary view
  decodes twice.
- **Fix for the decode:** reuse the compute kernel's 3-D geometry, or use a persistent
  grid-stride loop. That also cuts the partials from about 65k to a few hundred.
- **Too many stream operations:** a split reduction costs about 8 of them plus NCCL. Options:
  - a merge kernel that reads both partials directly;
  - a "last block done" finish instead of the CUB launch;
  - batching several scalars into one `ncclAllReduce` (CG needs two per iteration).

  For batching: the API takes a `count`, but `reduce_stages.h:111` hard-codes 1.
- **Boundary index math:** `set_and_validate(Boundary)` rebuilds `shell_bands_()` twice per
  thread, with 64-bit division (`span.h:441-443,478,499`). Precompute the band prefix sums on
  the host, use 32-bit indices below 2³¹ cells, and template kernels on `DataView`.

### 3.8 Process-grid choice ignores the domain

- **Where:** `src/process_grid.cpp:60-77` assigns prime factors greedily without looking at
  the extent.
- **What:** prime rank counts requested as Brick or Pencil silently become slabs, while
  `mode()` still reports Brick/Pencil. Brick with 4 ranks gives a pencil `(1,2,2)`.
- **Fix:** add an extent-aware `make(world, mode, global_extent)` that minimises halo
  surface. Log the chosen dims on rank 0.
- **Related:** for non-legacy slabs the z halo uses the isotropic radius (`layout.h:78-84`);
  use `radius3.z`.

## 4. Tests, CI and benchmarks

### 4.1 The distributed GPU test proves little today (verified)

- **Where:** `tests/test_dense_distributed_dist.cu`, registered with 4 ranks
  (`tests/CMakeLists.txt:455`).
- **Silent skips (`:717-725`):** it returns 0 when world is not 4 or 8, or when
  `device_count() < world`. That covers 2-GPU development machines, and also per-rank
  pinning (`srun --gpus-per-task=1`), where each rank sees one device. The 8-rank brick
  (2,2,2) is never registered.
- **Stops at the first failure (`:766-777`):** on the 8-GPU node it fails on the third case,
  `slab-periodicX`, so no pencil or brick case has run on a GPU.
- **The failure is a wrong expectation, not a library bug:**
  - On a periodic axis split over one rank, the run-based path copies the wrap inside the
    rank. With extent (9,8,11) over (1,1,4) that is exactly 80 intra pairs on ranks 0–2 and
    64 on rank 3, which is what the log shows.
  - The packed path stages the same wrap with kernels and reports 0, so only `/runs` fails.
  - `pencil-periodicX` would fail next in the same way.
  - The library side: `pair_count()` / `dist_entry_count()` mean different things in packed
    and run-based modes, and their doc (`halo_distributed.h:190-230`) is wrong for both.
- **Per-rank checks can hang:** a failing rank returns while its peers block in NCCL. MPI
  tests have no `TIMEOUT`, no `PROCESSORS` and no `RESOURCE_LOCK`. The comment at
  `tests/CMakeLists.txt:113` says the lock protects NCCL tests, but it isn't set for them.
- **The library's own header contradicts the test:** `halo_distributed.h:42-43` says
  periodic wrap-around copies inside a rank go through the intra-process `Transfer`, so the
  test asserts something the library documents as false.
- **Fix:**
  - correct the expectation;
  - run all cases and fail at the end;
  - combine `ok` across ranks with an allreduce;
  - gate only on `world_size()`;
  - honour `CARBON_TEST_STRICT`, and report skips with CTest's `SKIP_RETURN_CODE` instead of
    returning 0;
  - add a 2-rank case set (`(1,1,2)`, `(1,2,1)`, `(2,1,1)`, each open and periodic) so 2-GPU
    machines also cover X/Y exchange;
  - register an 8-rank variant;
  - give MPI tests `TIMEOUT`, `PROCESSORS` and `RESOURCE_LOCK`.

### 4.2 Coverage gaps

The following have no test at all:
- reductions on `DenseDistributed`;
- the host path (`make_host_kernel`, `HaloDistributed::launch_on_host`);
- `Extended` / `TwoWayExtended` overlap on pencil/brick;
- running a Skeleton more than once, so iteration k → k+1 hazards are invisible;
- non-`Sum` reductions;
- a z-only stencil on pencil/brick (§1.2);
- radius 2 on GPU;
- a real D3Q19 `LatticeDistributed` exchange on GPU (edges/corners are only checked in the
  host emulation);
- ranks that own zero cells;
- one rank with an invalid chunk (§1.1);
- prime rank counts in `ProcessGrid`;
- single-process multi-GPU `DenseDistributed` slab (`src/dense_distributed.cpp:185-192`),
  which LBM and CG use when launched without `mpiexec`;
- the XML VTK writer at host level (§5.2);
- distributed checkpoint (unsupported, §5.4).

Suggested first addition: a ping-pong stencil plus norm test, over all overlap modes, at least
3 iterations, on slab, pencil and brick.

### 4.3 Benchmarks

- **LBM distributed validates nothing:**
  - `--checksum` only prints the total mass, which is a weak invariant;
  - the help text claims bit-comparable results, but the runbook says about 1e-7;
  - fix: add `--validate` against a single-GPU `Dense` run at small N, or against a
    Ghia-style centreline profile, and register a 4-rank smoke test (N=32, 50 iterations).
- **LBM timing:**
  - there is no barrier around the timed window;
  - only rank 0's clock is reported, and the "already reduced across ranks" comment is
    wrong;
  - `--report-every` drains the streams inside the window;
  - fix: barrier, then max elapsed across ranks, and report min/max in the JSON.
- **LBM is a stale fork:**
  - copied before #73, so it has no fp64, no `--reps` and no `bench_report.h`;
  - it emits the same `carbon-lbm-report-v1` schema name with different fields;
  - the `Lattice` → `LatticeDistributed` selection (`src/lbm_kernels.h:254-267`) is
    correctness policy that belongs in the library;
  - fix: fold `--grid dense_distributed --decomposition` into the main benchmark.
- **CG (`benchmarks/cg/cg_poisson.cu`):**
  - the header promises an RMS check against the ramp that doesn't exist;
  - the exit code differs between ranks at `max_iter`: rank 0 reports success even without
    convergence, while the other ranks report failure;
  - direct `nvtxRangePushA` calls bypass the guarded macros (likely breaks HIP-AMD / SYCL);
  - it is a 1,170-line fork of the 836-line example.

### 4.4 Build and CI

- **Unpinned NCCL (verified):** `docker/Dockerfile.cuda:18,25` (also `.sycl` and
  `.hip-nvidia`) runs `--allow-change-held-packages` with unpinned `libnccl2 libnccl-dev`.
  That replaced the base image's CUDA-12.6 NCCL with 2.30 / CUDA 13.2, so on driver 535 every
  MPI test fails at `ncclCommInitRank`. Pin `NCCL_VERSION`, and pin base images by digest.
- **No CUDA architecture (verified):** the `cuda` / `cuda-mpi` configs in
  `scripts/docker-build-all.sh` never pass `CMAKE_CUDA_ARCHITECTURES`, so kernels rely on PTX
  JIT. `AGENTS.md` itself warns against this.
- **Unformatted files:** ten #76 files fail `format-check`, including all three new tests, so
  the PR likely merged with the format lane red or not required. Make `format-check` and a
  4-GPU distributed lane required checks.
- **Single-process tests break with more GPUs:** `skeleton_launch_order` and
  `reduce_fused_stencil` abort with 8 visible GPUs (z = 1 per slab). Cap them at
  `min(device_count, 4)` GPUs, or scale z.
- **Root `Dockerfile`:** it runs `ctest` inside `docker build` (no GPU, no MPI, no
  architecture). Fix or delete it.

## 5. I/O and checkpoint

### 5.1 VTK shards leave gaps and drop physical coordinates (verified)

- **Where:** `src/io_vtk_distributed.cpp:720-728,1083-1089`.
- **Gap:** each shard writes `PointData` with extent `0..n-1` and `Origin` at its first owned
  cell, so between two shards there is one unit with no cell. ParaView slices, contours,
  volume rendering and streamlines all break at every rank boundary, on every split axis.
- **Coordinates:** `Spacing="1 1 1"` always. The physical origin and spacing that
  `DenseDistributed` takes are deliberately ignored (`detail/io_vtk_distributed.inl:30-32`),
  so a CG dump with spacing h comes out in index units.
- **Fix:** write `CellData` with extent `0..n` and `Origin = origin + spacing·offset`, which
  needs no halo data and fits cell-centred LBM/FD values. Or overlap point data by one plane,
  which needs fresh halos.

### 5.2 The shipped XML writer has no host test (verified by reading)

- **What:** Part D of `test_dense_distributed_host.cpp` drives the legacy big-endian helpers,
  which `VtkBoxSession` no longer calls. The XML path (base64, empty shards) runs only in the
  4/8-rank GPU test, with one dtype and no empty rank.
- **Fix:** test the XML helpers on the host across dtypes, 1–4 components, all three base64
  tail lengths, empty shards and corrupt input. Then delete the ~400 lines of legacy code.

### 5.3 Failure handling

- `finalize()` checks `good()` without flushing first (`:1412-1419`, and the `.vtm` at
  `:1533`), so a full disk at close time goes unreported.
- A throw mid-payload still writes the closing tags, which leaves a truncated file that looks
  complete.
- Writes truncate the target in place, so a crash also destroys the previous output.
- Rank 0 writes the `.vtm` without knowing whether the other shards succeeded.
- **Fix:** flush, close and check; skip the footer on error; write to `.tmp` and rename;
  agree on success across ranks (§2) before rank 0 writes the index.

### 5.4 Scalability and restart

- **Scalability, write side:**
  - no gather to rank 0, which is good;
  - the base64 encoder writes 4 characters per `ostream::write`, about 500 M calls for
    1.6 GB;
  - one file per rank per dump;
  - "raw appended was rejected by VTK 9.2" is worth re-testing, since ParaView writes
    multi-GB raw-appended `.vti` routinely, so it was more likely an offset bug;
  - longer term: `.pvti` with raw appended data, or VTKHDF over the existing parallel HDF5.
- **Scalability, read side:** `parse_vti_box_header` reads each whole one-line payload with
  `getline` just to skip it, then decodes one character per `get()`.
- **No distributed checkpoint:**
  - `DenseDistributed` checkpoint/restart doesn't compile (it is `Dense`-only), so it fails
    safely;
  - the VTK reader allows at most 4 components, so LBM populations (Q = 19/27) cannot be
    restarted;
  - add box-aware native or HDF5 checkpointing, and raise the component limit.

## 6. API, maintainability, docs

- **Duplication with `Dense`:**
  - after renaming, the CUDA kernel, launch and reduce-kernel files and all three SYCL files
    are byte-identical to the `Dense` ones (about 800 lines);
  - the `DenseDistributedPolicy` arms in `launch_traits.h` are the same as the `else` arms;
  - `detail/dense_distributed/field.h` is a near copy of `detail/dense/field.h`;
  - this is why §1.3 and §1.4 exist in four places.

  Since all multi-axis logic lives in `Span`, one templated kernel set is possible. Longer
  term, `Dense` (single-process slabs) could become a facade over `DenseDistributed`.
- **Missing accessors:** `global_slab_count`, `local_z`, `z_offset_global` and similar are
  z-only and misleading for pencil/brick, and there is no `owned_offset()` /
  `local_coords()`.
- **Counters:** `pair_count()` / `dist_entry_count()` need clear semantics (§4.1). A
  `self_wrap_count()` would help.
- **Stale comments:**
  - `dense_distributed.cpp:196-197` (Stage-A Boundary);
  - `cudalike/dense_distributed_kernel.h:41-43` ("only z shifted");
  - `field.h:41-42` and `:465`;
  - the LBM `--vtk-out` help ("legacy, Dense-only");
  - the `.pvti` note in `io_vtk.inl`;
  - the CHANGELOG 2026-11 VTK entry, superseded later;
  - `AGENTS.md` test counts (26 / 116).
- **No user documentation for the distributed API.** `AGENTS.md` and the READMEs don't cover:
  - `DenseDistributed`, `ProcessGrid`, `BoundaryModeDistributed` or `HaloDistributed`;
  - how to launch and pin ranks to GPUs;
  - `CARBON_HALO_PACK`, `CARBON_HALO_PACK_Z` or `CARBON_HALO_PASS_BARRIER`;
  - the rule that single-process multi-GPU pencil/brick is rejected. The LBM non-MPI path
    builds exactly that.

  A "Distributed grids" page with a launch recipe and the support matrix would close this.

## 7. What is well designed

- **Pure host-side layout and plan code, shared verbatim with the kernels.** `layout.h` and
  the pack plans are host+device functions. `test_dense_distributed_host.cpp` runs the real
  layout and plan code on emulated worlds:
  - 9 grids × 6 boundary modes × 8 scenarios;
  - uneven splits, radius 2, D3Q19/D3Q27;
  - checked against independent ground truth.

  This is the strongest part of the PR.
- **Staged X → Y → Z passes are correct by construction.** Edges and corners come out exact
  with no diagonal messages. Message keys are symmetric across ranks, including the case where
  the same peer sits on both sides. Staging buffers and NCCL entry lists are built once.
- **Kernels don't know the decomposition.** All pencil/brick logic is in `Span`. The compact
  Boundary shell keeps the Boundary thread count close to the shell cell count, and
  Internal + Boundary = Standard is tested.
- **Clear contracts:**
  - Z-fastest rank mapping, which keeps the cheapest exchange node-local;
  - a separate `BoundaryModeDistributed` type;
  - strict `ProcessGrid::custom` validation;
  - empty-rank handling in launch and VTK;
  - clean rejection of unsupported multi-GPU-per-rank configurations.
- **The reduction pipeline merges Internal and Boundary before the single NCCL allreduce.** It
  is enqueued asynchronously, with no host sync on CUDA.

## 8. Suggested order of work

1. **Small, high-value fixes (about a day):**
   - §1.1 global validation;
   - §1.2 Boundary guard;
   - §1.3 HIP atomics;
   - §1.4 reject y > 1;
   - NCCL RAII group guard;
   - the `%%` fix;
   - test expectation, run-all-cases, and MPI `TIMEOUT`/`RESOURCE_LOCK`;
   - pin NCCL and the CUDA architecture in Docker;
   - clang-format the #76 files.
2. **Make the distributed path trustworthy:**
   - an `all_ok` collective error helper;
   - the ping-pong stencil plus norm test on slab/pencil/brick × overlap modes;
   - an LBM `--validate` smoke test in ctest;
   - fix the LBM timing.
3. **Performance, measured against the `hp-halo-ab/` baseline:**
   - fused single-group halo for face-only stencils, and no corner columns for D3Q19;
   - per-pass single pack launch, with an X-face thread mapping;
   - padded row pitch (opt-in) and a separate brick x-face Boundary launch;
   - graph capture of the iteration;
   - a halo communicator, if nsys shows ordering behind allreduce.
4. **I/O:** cell-data shards with physical coordinates; flush/atomic/collective failure
   handling; host XML tests; then distributed checkpoint (HDF5 hyperslabs).
5. **Structure:**
   - merge the duplicated `Dense` / `DenseDistributed` backends;
   - fold the LBM fork into one benchmark;
   - move the semantic choice into the library;
   - write the distributed user docs.
