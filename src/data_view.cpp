#include <carbon/data_view.h>

#include <carbon/logging.h>

#include <algorithm>
#include <cctype>
#include <ostream>
#include <stdexcept>
#include <string>

namespace carbon {

namespace {

constexpr std::string_view kStandardName{"standard"};
constexpr std::string_view kInternalName{"internal"};
constexpr std::string_view kBoundaryName{"boundary"};

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
to_string(DataView v) -> std::string_view {
  switch (v) {
    case DataView::Standard:
      return kStandardName;
    case DataView::Internal:
      return kInternalName;
    case DataView::Boundary:
      return kBoundaryName;
  }
  log::throw_with_log<std::out_of_range>("carbon::to_string(DataView): unknown enum value {}",
                                         static_cast<int>(v));
}

auto
from_string(std::string_view s) -> DataView {
  const std::string lower = to_lower_ascii(s);
  if (lower == kStandardName) {
    return DataView::Standard;
  }
  if (lower == kInternalName) {
    return DataView::Internal;
  }
  if (lower == kBoundaryName) {
    return DataView::Boundary;
  }
  log::throw_with_log<std::out_of_range>("carbon::from_string(DataView): unknown name '{}'", s);
}

auto
to_int(DataView v) noexcept -> int {
  return static_cast<int>(v);
}

auto
from_int(int v) -> DataView {
  if (v < 0 || v >= static_cast<int>(data_view_count)) {
    log::throw_with_log<std::out_of_range>(
        "carbon::from_int(DataView): value {} out of range [0, {})", v, data_view_count);
  }
  return static_cast<DataView>(v);
}

auto
data_view_options() noexcept -> std::array<DataView,
                                           data_view_count> {
  return {DataView::Standard, DataView::Internal, DataView::Boundary};
}

auto
operator<<(std::ostream& os,
           DataView v) -> std::ostream& {
  return os << to_string(v);
}

}  // namespace carbon
