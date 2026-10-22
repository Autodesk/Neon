#include <carbon/carbon.h>
#include <carbon/logging.h>

#include <cassert>

int
main(int argc,
     char** argv) {
  (void)argc;
  (void)argv;

  const int device_count = carbon::device_count();
  carbon::init();
  if (device_count == 0) {
    CARBON_LOG_ERROR("test_gpu_group_all_devices: no CUDA devices reported (cudaGetDeviceCount)");
    return 1;
  }

  const carbon::detail::GpuGroup gpu_group(carbon::get_all_devices());
  assert(gpu_group.size() == static_cast<size_t>(device_count));
  for (int i = 0; i < device_count; ++i) {
    assert(gpu_group.devices()[static_cast<size_t>(i)].device_id() == i);
  }
  return 0;
}
