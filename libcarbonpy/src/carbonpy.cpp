#include <carbonpy/carbonpy.h>

#include <carbon/carbon.h>
#include <carbon/devices.h>
#include <carbon/system.h>

#include <memory>
#include <mutex>

namespace {

std::once_flag g_init_once;
bool g_initialized = false;

}  // namespace

extern "C" auto
carbonpy_init() -> int {
  std::call_once(g_init_once, []() {
    carbon::init();
    g_initialized = true;
  });
  return 0;
}

extern "C" auto
carbonpy_shutdown() -> int {
  return 0;
}

extern "C" auto
carbonpy_device_count() -> int {
  carbonpy_init();
  return carbon::device_count();
}

extern "C" auto
carbonpy_system_new(void** handle) -> int {
  if (handle == nullptr) {
    return -1;
  }
  carbonpy_init();
  try {
    auto devices = carbon::get_all_devices();
    auto sys = std::make_unique<carbon::System>(carbon::detail::GpuGroup{std::move(devices)});
    *handle = sys.release();
    return 0;
  } catch (...) {
    return -1;
  }
}

extern "C" auto
carbonpy_system_delete(void** handle) -> int {
  if (handle == nullptr || *handle == nullptr) {
    return -1;
  }
  delete static_cast<carbon::System*>(*handle);
  *handle = nullptr;
  return 0;
}
