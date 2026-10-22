#include <carbon/carbon.h>
#include <carbon/logging.h>

#include <cstdlib>
#include <string_view>

/**
 * @file
 *
 * Environment gate for the @c needs_multi_device test class.
 *
 * Roughly two dozen tests in this suite open with
 * @code if (device_count() < 2) { CARBON_LOG_INFO("...skipping"); return 0; } @endcode
 * so on a single-GPU machine they report a pass without executing the cross-device peer
 * copies, halo seams, multi-slab layouts, and multi-device graph captures they exist to
 * cover. A green suite therefore means something materially weaker on one GPU than on two,
 * and nothing in CTest's output distinguishes the two situations.
 *
 * This gate turns that implicit downgrade into an explicit result:
 *
 *   - Two or more visible devices: pass.
 *   - Fewer than two and @c CARBON_TEST_STRICT unset: pass, but log a warning that names
 *     the consequence, so a local run still reads honestly.
 *   - Fewer than two and @c CARBON_TEST_STRICT=1: fail. A CI lane that is provisioned for
 *     multi-GPU hardware sets this, so a mis-provisioned runner turns red instead of
 *     silently reducing every @c needs_multi_device test to a no-op.
 *
 * A single gate is deliberate. Teaching all ~24 skip sites to consult the environment would
 * duplicate one policy decision two dozen times and drift the moment a multi-GPU test is
 * added. @c ctest @c -L @c needs_multi_device lists the tests whose coverage this gate speaks
 * for.
 *
 * Querying the device count must not itself fail the gate: on a driverless host the count
 * is simply zero, which is the case this test is here to report on.
 */

namespace {

[[nodiscard]] auto
strict_mode_requested() -> bool {
  const char* raw = std::getenv("CARBON_TEST_STRICT");
  if (raw == nullptr) {
    return false;
  }
  const std::string_view value{raw};
  return !(value.empty() || value == "0" || value == "false" || value == "off" || value == "OFF" ||
           value == "FALSE");
}

[[nodiscard]] auto
visible_device_count() -> int {
  try {
    carbon::init();
    return carbon::device_count();
  } catch (...) {
    return 0;
  }
}

}  // namespace

int
main() {
  constexpr int kRequired = 2;

  const int devices = visible_device_count();
  const bool strict = strict_mode_requested();

  if (devices >= kRequired) {
    CARBON_LOG_INFO(
        "test_multi_gpu_gate: {} visible devices; the needs_multi_device tests run their "
        "cross-device coverage.",
        devices);
    return 0;
  }

  if (strict) {
    CARBON_LOG_ERROR(
        "test_multi_gpu_gate: CARBON_TEST_STRICT is set but only {} device(s) are visible. "
        "Every test labelled needs_multi_device self-skips below {} devices, so this run cannot "
        "substantiate any cross-device claim (peer copy, halo seam, multi-slab layout, "
        "multi-device graph capture). Provision a second GPU, or unset CARBON_TEST_STRICT to "
        "accept the reduced coverage.",
        devices, kRequired);
    return 1;
  }

  CARBON_LOG_WARN(
      "test_multi_gpu_gate: only {} device(s) visible. Tests labelled needs_multi_device will "
      "self-skip and report a pass they did not earn — this run proves nothing about "
      "cross-device behaviour. Set CARBON_TEST_STRICT=1 to make that a failure instead.",
      devices);
  return 0;
}
