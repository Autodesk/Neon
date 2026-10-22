// System::default_stream() — the shared main/anchor stream.
//
// Pins the three properties the anchor contract rests on:
//   1. Shape: one per-GPU stream, same width as gpu_group().
//   2. Identity: every call returns the SAME underlying stream, and a shallow copy of
//      the System (the form Dense / Field hold internally) observes that same stream.
//      This is what lets two independent orchestrators share one FIFO — the property
//      Skeleton's start/end fork-join anchor depends on.
//   3. Distinctness from stream(): stream() still mints a fresh stream per call, so it
//      must not alias the default one.
//
// Also checks the empty-System case stays device-free (empty stream, no driver call).

#include <carbon/carbon.h>
#include <carbon/devices.h>
#include <carbon/logging.h>
#include <carbon/system.h>

#include <cstdlib>

int
main() {
  carbon::init();
  if (carbon::device_count() < 1) {
    CARBON_LOG_WARN("test_system_default_stream: no visible CUDA devices; skipping");
    return 0;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();

  // 1. Shape.
  const carbon::Stream def = sys.default_stream();
  if (def.size() != n_gpus) {
    CARBON_LOG_ERROR("test_system_default_stream: default_stream().size()={} != gpu_group()={}",
                     def.size(), n_gpus);
    return 1;
  }

  // 2. Identity across calls and across shallow copies. get_uid() is the address of the
  // shared per-GPU stream vector, so equal uids mean "the same streams", on any backend
  // (StreamGroup::native_at is CUDA-only).
  const carbon::Stream def2 = sys.default_stream();
  const carbon::System sys_copy = sys;
  const carbon::Stream def_copy = sys_copy.default_stream();
  if (def.get_uid() != def2.get_uid()) {
    CARBON_LOG_ERROR("test_system_default_stream: default_stream() not stable across calls");
    return 1;
  }
  if (def.get_uid() != def_copy.get_uid()) {
    CARBON_LOG_ERROR("test_system_default_stream: System copy sees a different default stream");
    return 1;
  }

  // 3. A fresh stream() must be a genuinely different stream.
  const carbon::Stream fresh = sys.stream();
  if (fresh.size() != n_gpus) {
    CARBON_LOG_ERROR("test_system_default_stream: stream().size()={} != gpu_group()={}",
                     fresh.size(), n_gpus);
    return 1;
  }
  if (fresh.get_uid() == def.get_uid()) {
    CARBON_LOG_ERROR("test_system_default_stream: stream() aliases default_stream()");
    return 1;
  }

  // 4. The device-free placeholder stays empty (and must not throw).
  const carbon::System empty;
  if (empty.default_stream().size() != 0U) {
    CARBON_LOG_ERROR("test_system_default_stream: empty System default_stream() is not empty");
    return 1;
  }

  CARBON_LOG_INFO("test_system_default_stream: ok ({} GPUs)", n_gpus);
  return 0;
}
