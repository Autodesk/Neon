#include <carbon/detail/dense/launch_shape.h>
#include <carbon/detail/gpu.h>

#include <algorithm>

namespace carbon {

auto
dense_make_kernel_block(const detail::Gpu& gpu) noexcept -> LaunchShape {
  static std::mutex mu;
  static std::unordered_map<int, LaunchShape> cache;

  const int device_id = gpu.device_id();
  {
    std::lock_guard<std::mutex> lock(mu);
    const auto it = cache.find(device_id);
    if (it != cache.end()) {
      return it->second;
    }
  }

  const detail::Properties& props = gpu.properties();
  int target = props.max_threads_per_block;
  if (props.max_threads_per_multiprocessor > 0) {
    target = std::min(target, props.max_threads_per_multiprocessor);
  }
  target = std::min(target, 512);
  if (props.warp_size > 0) {
    target = (target / props.warp_size) * props.warp_size;
  }
  const LaunchShape block = dense_factor_block_threads(target);

  std::lock_guard<std::mutex> lock(mu);
  cache[device_id] = block;
  return block;
}

}  // namespace carbon
