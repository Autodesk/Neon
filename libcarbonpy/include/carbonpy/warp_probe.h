#pragma once

// Spike probe (phase 3 of the two-layer binding plan). Answers one question that gates the
// design of *both* layers: can a Warp builtin return a registered external type that itself
// nests another registered external type?
//
// Layer 1 needs it for `set_and_validate` returning C++'s `ValidatedIdx { Idx idx; bool valid; }`;
// layer 2 needs it for the fat index `{ Idx idx; Span span; }` that lets XLB kernel bodies stay
// verbatim. If Warp's codegen refuses, layer 1 falls back to `make_idx` plus a separate validity
// call and layer 2 threads the span explicitly.
//
// Four escalating capabilities are probed, because they are genuinely distinct:
//   1. registering a ctypes struct that nests two already-registered types;
//   2. returning it by value from a builtin;
//   3. passing that value back into another builtin, which reads the nested members in C++;
//   4. an accessor builtin returning a *nested* registered type back out to the kernel.
// (2) and (4) are the unproven ones; nesting on the *input* side is already shipping, since
// CarbonDensePartition embeds a CarbonDenseSpan and CarbonMgridPartition embeds three spans.

#include "carbonpy/warp_dense.h"

namespace wp {

struct CarbonProbeFatIdx {
  CarbonDenseIdx idx;
  CarbonDenseSpan span;

  CARBON_WARP_HD CarbonProbeFatIdx() = default;

  CARBON_WARP_HD CarbonProbeFatIdx(const char* bytes,
                                   size_t n) {
    if (n == sizeof(*this)) {
      memcpy(this, bytes, n);
    }
  }
};

struct CarbonProbeValidatedIdx {
  CarbonDenseIdx idx;
  int32_t valid;

  CARBON_WARP_HD CarbonProbeValidatedIdx() = default;

  CARBON_WARP_HD CarbonProbeValidatedIdx(const char* bytes,
                                         size_t n) {
    if (n == sizeof(*this)) {
      memcpy(this, bytes, n);
    }
  }
};

namespace carbon {

CARBON_WARP_HD auto
probe_make_fat(const CarbonDenseSpan& span,
               int64_t ix,
               int64_t iy,
               int64_t iz) -> CarbonProbeFatIdx {
  CarbonProbeFatIdx out{};
  out.idx = carbon_dense_set(span, ix, iy, iz);
  out.span = span;
  return out;
}

CARBON_WARP_HD auto
probe_fat_read(const CarbonDensePartitionF32& part,
               const CarbonProbeFatIdx& cell,
               int component_id) -> float {
  return carbon_dense_partition_read_f32(part, cell.span, cell.idx, component_id);
}

CARBON_WARP_HD auto
probe_fat_write(const CarbonDensePartitionF32& part,
                const CarbonProbeFatIdx& cell,
                int component_id,
                float value) -> void {
  carbon_dense_partition_write_f32(part, cell.span, cell.idx, component_id, value);
}

CARBON_WARP_HD auto
probe_fat_idx(const CarbonProbeFatIdx& cell) -> CarbonDenseIdx {
  return cell.idx;
}

CARBON_WARP_HD auto
probe_set_and_validate(const CarbonDenseSpan& span,
                       int64_t ix,
                       int64_t iy,
                       int64_t iz) -> CarbonProbeValidatedIdx {
  CarbonProbeValidatedIdx out{};
  out.idx = carbon_dense_set(span, ix, iy, iz);
  out.valid = (ix >= 0 && iy >= 0 && iz >= 0 && ix < span.logic_x && iy < span.logic_y &&
               iz < span.logic_z)
                  ? 1
                  : 0;
  return out;
}

CARBON_WARP_HD auto
probe_validated_valid(const CarbonProbeValidatedIdx& v) -> bool {
  return v.valid != 0;
}

CARBON_WARP_HD auto
probe_validated_idx(const CarbonProbeValidatedIdx& v) -> CarbonDenseIdx {
  return v.idx;
}

}  // namespace carbon
}  // namespace wp
