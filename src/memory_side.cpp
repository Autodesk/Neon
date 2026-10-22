#include <carbon/memory_side.h>

#include <carbon/logging.h>

#include <algorithm>
#include <cctype>
#include <ostream>
#include <stdexcept>
#include <string>

namespace carbon {

namespace {

constexpr std::string_view kHostName{"host"};
constexpr std::string_view kDeviceName{"device"};
constexpr std::string_view kHostDeviceCanonical{"host_device"};
// Aliases that callers might reasonably type — normalised in memory_side_from_string.
constexpr std::string_view kHostDeviceCompact{"hostdevice"};
constexpr std::string_view kHostDeviceHyphen{"host-device"};

auto
to_lower_ascii(std::string_view s) -> std::string {
  std::string out;
  out.resize(s.size());
  std::transform(s.begin(), s.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

}  // namespace

auto
to_string(MemorySide v) -> std::string_view {
  switch (v) {
    case MemorySide::Host:
      return kHostName;
    case MemorySide::Device:
      return kDeviceName;
    case MemorySide::HostDevice:
      return kHostDeviceCanonical;
  }
  log::throw_with_log<std::out_of_range>("carbon::to_string(MemorySide): unknown enum value {}",
                                         static_cast<int>(v));
}

auto
memory_side_from_string(std::string_view s) -> MemorySide {
  const std::string lower = to_lower_ascii(s);
  if (lower == kHostName) {
    return MemorySide::Host;
  }
  if (lower == kDeviceName) {
    return MemorySide::Device;
  }
  if (lower == kHostDeviceCanonical || lower == kHostDeviceCompact || lower == kHostDeviceHyphen) {
    return MemorySide::HostDevice;
  }
  log::throw_with_log<std::out_of_range>("carbon::memory_side_from_string: unknown name '{}'", s);
}

auto
to_int(MemorySide v) noexcept -> int {
  return static_cast<int>(v);
}

auto
memory_side_from_int(int v) -> MemorySide {
  if (v < 0 || v >= static_cast<int>(memory_side_count)) {
    log::throw_with_log<std::out_of_range>(
        "carbon::memory_side_from_int: value {} out of range [0, {})", v, memory_side_count);
  }
  return static_cast<MemorySide>(v);
}

auto
memory_side_options() noexcept -> std::array<MemorySide,
                                             memory_side_count> {
  return {MemorySide::Host, MemorySide::Device, MemorySide::HostDevice};
}

auto
operator<<(std::ostream& os,
           MemorySide v) -> std::ostream& {
  return os << to_string(v);
}

auto
has_device(MemorySide v) noexcept -> bool {
  return v == MemorySide::Device || v == MemorySide::HostDevice;
}

auto
has_host(MemorySide v) noexcept -> bool {
  return v == MemorySide::Host || v == MemorySide::HostDevice;
}

}  // namespace carbon
