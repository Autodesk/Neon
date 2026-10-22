// End-to-end tutorial for carbon::CheckpointWriter / carbon::CheckpointReader on the
// Dense grid family. Walks through the canonical "checkpoint a set of fields" workflow:
//
//   Phase 1.  Build a Dense + a couple of Field<T, Components> instances.
//   Phase 2.  Seed them: one host-only field via fill_owned_host (scalar f32 "rho"),
//             one HostDevice field via a tiny init kernel (3-component i32 "u"). The
//             second field's authoritative copy lives on device after the kernel,
//             which is the realistic shape of "I just ran a step on the GPU and want
//             to checkpoint".
//   Phase 3.  Save: CheckpointWriter w("ckpt_step_50", dense, step=50, time=2.5);
//             w.sync_from_device(true).add("rho", rho).add("u", u).commit();
//             The sync_from_device flag tells the writer to issue a Transfer<T> from
//             gpu_buffers to cpu_buffers before serialising — exactly the right
//             behaviour for "u", a no-op for "rho" (it has no device copy).
//   Phase 4.  Inspect: print the resulting <dir>/manifest.json + per-field <name>.bin
//             files (or <name>.bin.rank<N> under MPI). The manifest is plain JSON, so
//             a future agent or post-processing script can parse it without linking
//             carbon.
//   Phase 5.  Resume: simulate a process restart by allocating fresh fields on the
//             same Dense layout. CheckpointReader r(dir, dense) parses the manifest,
//             cross-validates schema_version / global_extent / world_size, then
//             load("name", field) with sync_to_device(true) reads the per-rank shard
//             into cpu_buffers and pushes host->device on the way out.
//   Phase 6.  Verify: read the host mirror back and assert byte-for-byte parity with
//             the original seed. Prints "OK" on success, exits non-zero on mismatch.
//
// Distributed build (CARBON_ENABLE_MPI=ON, mpiexec -n N)
// ------------------------------------------------------
// The same source compiles and runs identically: ensure_directory mkdirs on rank 0
// and barriers, every rank writes its own <name>.bin.rank<N> shard inside add(), and
// commit() barriers, has rank 0 write manifest.json, and barriers again so every
// rank's CheckpointReader ctor sees the marker. No call-site changes; the only
// difference is what shows up on disk (one .bin.rank<N> file per rank per field).
//
// Build & run
// -----------
//   cmake --build build --target carbon_example_checkpoint_dense
//   ./build/examples/carbon_example_checkpoint_dense
//
// Distributed:
//   cmake --build build-mpi --target carbon_example_checkpoint_dense
//   mpiexec -n 2 ./build-mpi/examples/carbon_example_checkpoint_dense

#include <carbon/carbon.h>
#include <carbon/checkpoint.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Tiny wrappers around carbon::Distributed::{world_rank, world_size, barrier} that
// degrade to single-process defaults when carbon was built without MPI. Kept here
// (rather than included from carbon/checkpoint.h) so the tutorial source reads as a
// drop-in template — copy this file into your own project and the bootstrap works
// whether or not your build has MPI on.
[[nodiscard]] auto
world_rank(const carbon::System& sys) noexcept -> int {
#ifdef CARBON_HAVE_MPI_NCCL
  if (const auto* d = sys.distributed(); d != nullptr) {
    return d->world_rank();
  }
#else
  (void)sys;
#endif
  return 0;
}

[[nodiscard]] auto
world_size(const carbon::System& sys) noexcept -> int {
#ifdef CARBON_HAVE_MPI_NCCL
  if (const auto* d = sys.distributed(); d != nullptr) {
    return d->world_size();
  }
#else
  (void)sys;
#endif
  return 1;
}

void
barrier(const carbon::System& sys) {
#ifdef CARBON_HAVE_MPI_NCCL
  if (const auto* d = sys.distributed(); d != nullptr) {
    d->barrier();
  }
#else
  (void)sys;
#endif
}

// Deterministic per-cell encoders; both phases (save + verify) drive these so we can
// catch any byte-level corruption end-to-end.
[[nodiscard]] CARBON_HOST_DEVICE auto
encode_rho(int64_t x,
           int64_t y,
           int64_t global_z) noexcept -> float {
  return 0.125F * static_cast<float>(global_z * 10000 + y * 100 + x);
}

[[nodiscard]] CARBON_HOST_DEVICE auto
encode_u(int64_t x,
         int64_t y,
         int64_t global_z,
         int c) noexcept -> std::int32_t {
  return static_cast<std::int32_t>(global_z * 100000 + y * 1000 + x * 10 + c);
}

// Builds a tiny init kernel that writes encode_u(global_x, global_y, global_z, c)
// into every component of every owned cell of `u`. Demonstrates that the data the
// checkpoint persists is whatever lives on the GPU — sync_from_device pulls it back
// to the host mirror automatically.
[[nodiscard]] auto
build_u_init_kernel(const carbon::Dense& dense,
                    carbon::Dense::Field<std::int32_t,
                                         3>& u) -> carbon::Kernel {
  using Span = carbon::Dense::Span;
  using Idx = carbon::Dense::Idx;
  return carbon::Kernel::compute(
      dense,
      [&u](carbon::KernelLoader& loader) {
        const auto out = loader.write(u);
        return [out] CARBON_HOST_DEVICE(Span s, Idx idx) -> void {
          const auto gloc = s.global_location(idx);
          for (int c = 0; c < 3; ++c) {
            out.write(s, idx, c, encode_u(gloc.x, gloc.y, gloc.z, c));
          }
        };
      },
      "u_init");
}

// Print the first few KB of <dir>/manifest.json so the user can see what the writer
// emitted. Rank-0 only under MPI (the manifest is rank-0-only by design).
void
dump_manifest(const std::filesystem::path& dir,
              int rank) {
  if (rank != 0) {
    return;
  }
  const auto path = dir / "manifest.json";
  std::ifstream in(path);
  if (!in) {
    CARBON_LOG_WARN("checkpoint_dense: failed to open '{}' for inspection", path.string());
    return;
  }
  std::ostringstream oss;
  oss << in.rdbuf();
  CARBON_LOG_INFO("checkpoint_dense: ----- {} -----\n{}", path.string(), oss.str());
}

}  // namespace

int
main() {
  carbon::init();

  // Bootstrap: enumerate visible GPUs, build a System, build a Dense layout. Under
  // MPI we construct a carbon::Distributed and feed it to System so the per-rank
  // shards land in the right files; without MPI the System owns every visible GPU
  // on this node. The same source compiles + runs in either mode.
  if (carbon::device_count() < 1) {
    CARBON_LOG_ERROR("checkpoint_dense: no CUDA devices visible — aborting");
    return EXIT_FAILURE;
  }
#ifdef CARBON_HAVE_MPI_NCCL
  auto distributed = std::make_shared<const carbon::Distributed>();
  carbon::System sys = (distributed->world_size() > 1) ? carbon::System(distributed)
                                                       : carbon::System(carbon::get_all_devices());
#else
  carbon::System sys(carbon::get_all_devices());
#endif
  const size_t n_gpus = sys.gpu_group().size();
  const int rank = world_rank(sys);
  const int ranks = world_size(sys);

  // 7-point stencil so the Dense reserves a 1-thick z halo; we never touch halos in
  // this example (CheckpointReader / Writer act on owned cells only) but keeping the
  // halo size > 0 makes the layout realistic.
  carbon::Stencil stencil(std::vector<carbon::int32_3d>{
      {0, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}});

  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{4, 4, z_per_slab * static_cast<int64_t>(n_gpus)};
  carbon::Dense dense(sys, global, std::move(stencil));

  // Two fields with different shapes:
  //   - rho : scalar f32, MemorySide::Host (no device buffer; CheckpointWriter sees
  //           cpu_buffers populated by fill_owned_host, ignores sync_from_device for
  //           this one because there's nothing on device to flush).
  //   - u   : 3-component i32, MemorySide::HostDevice (device authoritative after
  //           the init kernel; sync_from_device pulls it back).
  auto rho = dense.newField<float, 1>(carbon::MemorySide::Host).set_name("rho");
  auto u = dense.newField<std::int32_t, 3>(carbon::MemorySide::HostDevice).set_name("u");

  rho.fill_owned_host([](carbon::int64_3d gc, int /*c*/) {
    return encode_rho(gc.x, gc.y, gc.z);
  });

  // Drive the init kernel through Skeleton so this works identically across
  // single-GPU / multi-GPU / cross-rank without any backend-specific scaffolding.
  // (Skeleton::compile auto-inserts halo updates when stencil reads are detected;
  // this kernel only writes, so there's nothing to insert.)
  carbon::Skeleton sk(sys);
  sk.add(build_u_init_kernel(dense, u));
  sk.compile();
  sk.run();
  sk.wait();

  // ---------------------------------------------------------------------------
  // Phase 3: save.
  // ---------------------------------------------------------------------------
  namespace fs = std::filesystem;
  const fs::path dir = "ckpt_step_50";
  if (rank == 0) {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
  barrier(sys);

  constexpr int64_t kStep = 50;
  constexpr double kSimTime = 2.5;
  {
    carbon::CheckpointWriter w(dir.string(), dense, kStep, kSimTime);
    // sync_from_device(true) tells add() to issue a Transfer<T> device->host before
    // serialising. Safe to leave on for fields without a device side; it short-
    // circuits when cpu_buffers is the only allocation.
    w.sync_from_device(true);
    w.add("rho", rho).add("u", u);
    // commit() seals the checkpoint by emitting <dir>/manifest.json. Without this,
    // CheckpointReader treats the directory as in-flight and throws on construction.
    w.commit();
  }
  CARBON_LOG_INFO("checkpoint_dense: rank {} wrote checkpoint at '{}' (step={}, time={})", rank,
                  dir.string(), kStep, kSimTime);

  // ---------------------------------------------------------------------------
  // Phase 4: inspect what landed on disk.
  // ---------------------------------------------------------------------------
  if (rank == 0) {
    CARBON_LOG_INFO("checkpoint_dense: directory contents:");
    for (const auto& entry : fs::directory_iterator(dir)) {
      CARBON_LOG_INFO("  {}  ({} bytes)", entry.path().filename().string(),
                      fs::file_size(entry.path()));
    }
    dump_manifest(dir, rank);
  }

  // ---------------------------------------------------------------------------
  // Phase 5: simulate a "fresh process" by allocating new fields.
  // ---------------------------------------------------------------------------
  // In a real restart you would build a fresh System / Dense from the same global
  // extent and stencil; we reuse the local ones for brevity. The reader does not
  // care whether the Dense it is given is the "same" one — it only checks that
  // global_extent + world_size match the manifest.
  auto rho_loaded = dense.newField<float, 1>(carbon::MemorySide::Host).set_name("rho_loaded");
  auto u_loaded =
      dense.newField<std::int32_t, 3>(carbon::MemorySide::HostDevice).set_name("u_loaded");

  carbon::CheckpointReader r(dir.string(), dense);
  CARBON_LOG_INFO("checkpoint_dense: reader sees step={}, time={}, fields=[{}]", r.step(), r.time(),
                  [&] {
                    std::ostringstream oss;
                    const auto names = r.field_names();
                    for (size_t i = 0; i < names.size(); ++i) {
                      oss << (i == 0 ? "" : ", ") << '\'' << names[i] << '\'';
                    }
                    return oss.str();
                  }());

  // sync_to_device(true) tells load() to issue a Transfer<T> host->device after the
  // host-side read so the reloaded field is immediately usable by a kernel without
  // a separate upload step. Safe for Host-only fields (the device side is empty so
  // the layer no-ops the upload through the same cpu_buffers / gpu_buffers guard).
  r.sync_to_device(true);
  r.load("rho", rho_loaded);
  r.load("u", u_loaded);

  // ---------------------------------------------------------------------------
  // Phase 6: verify owned-cell parity host-side.
  // ---------------------------------------------------------------------------
  size_t mismatches = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    const int64_t per_comp = span.linear_storage_elements();
    const float* rh = rho_loaded.cpu_buffers()[g].data();
    const std::int32_t* uh = u_loaded.cpu_buffers()[g].data();
    for (int64_t z = halo_z; z < halo_z + lim_z; ++z) {
      const int64_t global_z = z_origin + z;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          if (rh[off] != encode_rho(x, y, global_z)) {
            ++mismatches;
          }
          for (int c = 0; c < 3; ++c) {
            const int64_t voff = static_cast<int64_t>(c) * per_comp + off;
            if (uh[voff] != encode_u(x, y, global_z, c)) {
              ++mismatches;
            }
          }
        }
      }
    }
  }
  if (mismatches > 0U) {
    CARBON_LOG_ERROR("checkpoint_dense: rank {} verification FAILED ({} mismatches)", rank,
                     mismatches);
    return EXIT_FAILURE;
  }

  barrier(sys);
  if (rank == 0) {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
  CARBON_LOG_INFO("checkpoint_dense: rank {} of {} OK ({} GPU(s); rho f32 + u i32x3 round-trip)",
                  rank, ranks, n_gpus);
  return EXIT_SUCCESS;
}
