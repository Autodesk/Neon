#include <carbon/carbon.h>
#include <carbon/logging.h>
#include <carbon/version.h>

#include <cstring>

int
main() {
  carbon::init();
  const char* const version = carbon::info();
  if (version == nullptr) {
    CARBON_LOG_ERROR("test_dynamic_load: carbon::info() returned null");
    return 1;
  }
  // info() returns the full version string: "<MAJOR>.<MINOR>" on a tarball
  // build, "<MAJOR>.<MINOR> (<git-describe>)" on a git checkout. Match the
  // semantic-version prefix in both cases — the git-describe tail is
  // unstable across commits and is not what this test is locking.
  const char* const want_prefix = CARBON_VERSION_STRING;
  const size_t prefix_len = std::strlen(want_prefix);
  if (std::strncmp(version, want_prefix, prefix_len) != 0) {
    CARBON_LOG_ERROR("test_dynamic_load: unexpected version string: '{}' (want prefix '{}')",
                     version, want_prefix);
    return 2;
  }
  return 0;
}
