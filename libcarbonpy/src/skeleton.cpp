#include <carbon/data_view.h>
#include <carbon/skeleton.h>
#include <carbon/system.h>
#include <carbonpy/skeleton.h>
#include <carbonpy/warp_kernel.h>

#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

enum class ContainerKind : std::uint8_t {
  Dense = 0,
  Block = 1,
  Mgrid = 2,
};

struct PendingEntry {
  ContainerKind kind{ContainerKind::Dense};
  void* handle{};
};

auto
int_to_view(int v) -> carbon::DataView {
  switch (v) {
    case 1:
      return carbon::DataView::Internal;
    case 2:
      return carbon::DataView::Boundary;
    default:
      return carbon::DataView::Standard;
  }
}

auto
map_occ(int occ_mode) -> carbon::OverlapMode {
  switch (occ_mode) {
    case 0:
      return carbon::OverlapMode::Standard;
    case 1:
      return carbon::OverlapMode::Extended;
    case 2:
      return carbon::OverlapMode::TwoWayExtended;
    case 3:
      return carbon::OverlapMode::None;
    default:
      return carbon::OverlapMode::None;
  }
}

class CarbonSkeletonBox {
public:
  explicit CarbonSkeletonBox(carbon::System* system)
      : system_(system) {}

  auto
  set_occ(int occ_mode) -> void {
    occ_ = map_occ(occ_mode);
    skel_.reset();
  }

  auto
  clear() -> void {
    pending_.clear();
    skel_.reset();
  }

  auto
  add_dense(void* container) -> int {
    return add_container(ContainerKind::Dense, container);
  }

  auto
  add_block(void* container) -> int {
    return add_container(ContainerKind::Block, container);
  }

  auto
  add_mgrid(void* container) -> int {
    return add_container(ContainerKind::Mgrid, container);
  }

  auto
  compile() -> int {
    try {
      carbon::Skeleton::Options options{};
      options.overlap_mode = occ_;
      skel_.emplace(carbon::Skeleton(carbon::System(*system_), options));
      for (const PendingEntry& entry : pending_) {
        carbon::Kernel kernel;
        switch (entry.kind) {
          case ContainerKind::Dense:
            kernel = carbonpy::make_warp_dense_kernel(entry.handle);
            break;
          case ContainerKind::Block:
            kernel = carbonpy::make_warp_block_kernel(entry.handle);
            break;
          case ContainerKind::Mgrid:
            kernel = carbonpy::make_warp_mgrid_kernel(entry.handle);
            break;
        }
        skel_->add(std::move(kernel));
      }
      skel_->compile();
      return 0;
    } catch (...) {
      skel_.reset();
      return -1;
    }
  }

  auto
  run(int data_view) -> int {
    try {
      if (!skel_.has_value()) {
        const int rc = compile();
        if (rc != 0) {
          return rc;
        }
      }
      skel_->run(int_to_view(data_view));
      skel_->wait();
      return 0;
    } catch (...) {
      return -1;
    }
  }

  [[nodiscard]] auto
  inserted_halo_count() const -> int {
    if (!skel_.has_value() || !skel_->compiled()) {
      return 0;
    }
    int count = 0;
    for (const carbon::Kernel& kernel : skel_->effective_kernels()) {
      if (kernel.kind() == carbon::KernelKind::Halo) {
        ++count;
      }
    }
    return count;
  }

  [[nodiscard]] auto
  num_streams() -> int {
    if (ensure_compiled() != 0) {
      return -1;
    }
    return static_cast<int>(skel_->num_streams());
  }

  // Stream index assigned to the @p user_index -th user-added container. The C++
  // skeleton indexes kernel_streams() by effective-kernel order (user kernels +
  // auto-inserted halos), so map the user index back through effective_to_user().
  [[nodiscard]] auto
  kernel_stream_at(int user_index) -> int {
    if (user_index < 0 || ensure_compiled() != 0) {
      return -1;
    }
    const auto& effective_to_user = skel_->effective_to_user();
    const auto& streams = skel_->kernel_streams();
    for (size_t e = 0; e < effective_to_user.size(); ++e) {
      if (effective_to_user[e] == static_cast<size_t>(user_index)) {
        return static_cast<int>(streams[e]);
      }
    }
    return -1;
  }

  auto
  export_dot(const std::string& path,
             const std::string& graph_name) -> int {
    try {
      if (!skel_.has_value()) {
        const int rc = compile();
        if (rc != 0) {
          return rc;
        }
      }
      const std::string dot = skel_->export_dot(graph_name);
      std::ofstream out(path, std::ios::binary);
      if (!out) {
        return -1;
      }
      out << dot;
      return out.good() ? 0 : -1;
    } catch (...) {
      return -1;
    }
  }

  [[nodiscard]] auto
  user_kernel_count() const noexcept -> size_t {
    return pending_.size();
  }

private:
  // Compile on demand so read-only queries (num_streams / kernel_stream_at) work
  // right after add() without a separate compile() call. Idempotent: skips when a
  // compiled schedule already exists.
  auto
  ensure_compiled() -> int {
    if (skel_.has_value() && skel_->compiled()) {
      return 0;
    }
    return compile();
  }

  auto
  add_container(ContainerKind kind,
                void* container) -> int {
    if (container == nullptr) {
      return -1;
    }
    pending_.push_back(PendingEntry{kind, container});
    skel_.reset();
    return 0;
  }

  carbon::System* system_;
  carbon::OverlapMode occ_{carbon::OverlapMode::None};
  std::optional<carbon::Skeleton> skel_;
  std::vector<PendingEntry> pending_;
};

}  // namespace

extern "C" auto
carbonpy_skeleton_new(void** handle,
                      void* system_handle) -> int {
  if (handle == nullptr || system_handle == nullptr) {
    return -1;
  }
  try {
    auto* sys = static_cast<carbon::System*>(system_handle);
    *handle = new CarbonSkeletonBox(sys);
    return 0;
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_skeleton_delete(void** handle) -> int {
  if (handle == nullptr || *handle == nullptr) {
    return -1;
  }
  delete static_cast<CarbonSkeletonBox*>(*handle);
  *handle = nullptr;
  return 0;
}

extern "C" auto
carbonpy_skeleton_clear(void* skeleton_handle) -> int {
  if (skeleton_handle == nullptr) {
    return -1;
  }
  try {
    static_cast<CarbonSkeletonBox*>(skeleton_handle)->clear();
    return 0;
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_skeleton_add_dense_container(void* skeleton_handle,
                                      void* container_handle) -> int {
  if (skeleton_handle == nullptr) {
    return -1;
  }
  try {
    return static_cast<CarbonSkeletonBox*>(skeleton_handle)->add_dense(container_handle);
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_skeleton_add_block_container(void* skeleton_handle,
                                      void* container_handle) -> int {
  if (skeleton_handle == nullptr) {
    return -1;
  }
  try {
    return static_cast<CarbonSkeletonBox*>(skeleton_handle)->add_block(container_handle);
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_skeleton_add_mgrid_container(void* skeleton_handle,
                                      void* container_handle) -> int {
  if (skeleton_handle == nullptr) {
    return -1;
  }
  try {
    return static_cast<CarbonSkeletonBox*>(skeleton_handle)->add_mgrid(container_handle);
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_skeleton_compile(void* skeleton_handle,
                          int occ_mode) -> int {
  if (skeleton_handle == nullptr) {
    return -1;
  }
  try {
    auto* box = static_cast<CarbonSkeletonBox*>(skeleton_handle);
    box->set_occ(occ_mode);
    return box->compile();
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_skeleton_run(void* skeleton_handle,
                      int /*stream_idx*/,
                      int data_view) -> int {
  if (skeleton_handle == nullptr) {
    return -1;
  }
  try {
    return static_cast<CarbonSkeletonBox*>(skeleton_handle)->run(data_view);
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_skeleton_inserted_halo_count(void* skeleton_handle) -> int {
  if (skeleton_handle == nullptr) {
    return -1;
  }
  try {
    return static_cast<CarbonSkeletonBox*>(skeleton_handle)->inserted_halo_count();
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_skeleton_export_dot(void* skeleton_handle,
                             const char* path,
                             const char* graph_name) -> int {
  if (skeleton_handle == nullptr || path == nullptr) {
    return -1;
  }
  try {
    const std::string name = graph_name != nullptr ? graph_name : "carbon_skeleton";
    return static_cast<CarbonSkeletonBox*>(skeleton_handle)->export_dot(path, name);
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_skeleton_kernel_stream_count(void* skeleton_handle) -> int {
  if (skeleton_handle == nullptr) {
    return -1;
  }
  try {
    return static_cast<int>(
        static_cast<CarbonSkeletonBox*>(skeleton_handle)->user_kernel_count());
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_skeleton_kernel_stream_at(void* skeleton_handle,
                                   int index) -> int {
  if (skeleton_handle == nullptr) {
    return -1;
  }
  try {
    return static_cast<CarbonSkeletonBox*>(skeleton_handle)->kernel_stream_at(index);
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_skeleton_num_streams(void* skeleton_handle) -> int {
  if (skeleton_handle == nullptr) {
    return -1;
  }
  try {
    return static_cast<CarbonSkeletonBox*>(skeleton_handle)->num_streams();
  } catch (...) {
    return -1;
  }
}
