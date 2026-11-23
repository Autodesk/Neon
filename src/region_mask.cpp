#include <carbon/region_mask.h>

#include <string>

namespace carbon {

auto
to_string(const RegionMask& mask) -> std::string {
  if (mask.is_all()) {
    return "all";
  }
  if (mask.empty()) {
    return "none";
  }
  std::string out = "{";
  bool first = true;
  unsigned b = 0;
  while (b < RegionMask::kBits) {
    if (!mask.test(b)) {
      ++b;
      continue;
    }
    unsigned end = b;
    while (end + 1U < RegionMask::kBits && mask.test(end + 1U)) {
      ++end;
    }
    if (!first) {
      out += ',';
    }
    first = false;
    out += std::to_string(b);
    if (end > b) {
      out += '-';
      out += std::to_string(end);
    }
    b = end + 1U;
  }
  out += '}';
  return out;
}

}  // namespace carbon
