# LBM lid-driven cavity benchmark.
#
# Carbon port of Neon/benchmarks/lbm-lid-driven-cavity-flow. D3Q19 BGK, two-pop
# pull streaming, half-way bounce-back walls, moving lid on +y face. The
# iteration loop is wrapped in carbon::Skeleton so the halo exchange between
# successive collide-stream sweeps is auto-inserted (loader.read_stencil on the
# input population) — same pattern as examples/jacobi_unified.cu, scaled up
# from a 1-component scalar field to a 19-component population field.
#
# Headers live under src/ and are #included directly from the .cu / .cpp twin
# so a single TU drives the build. CLI parsing uses the bundled clipp header
# from third_party/ (same single-header parser examples/cg_poisson uses).

_carbon_benchmark_add(carbon_benchmark_lbm_lid_driven_cavity lbm_lid_driven_cavity.cu)
target_link_libraries(carbon_benchmark_lbm_lid_driven_cavity
  PRIVATE carbon::carbon carbon::third_party::clipp)
# Two include roots so the same `#include "src/cell_type.h"` headers in
# the .cu/.cpp twin resolve under both backends:
#   - CARBON_BACKEND=Cuda compiles the .cu in-place, so `src/...` already
#     resolves relative to the source dir; the explicit entry is redundant
#     but harmless.
#   - CARBON_BACKEND=Sycl configure_file's the .cu to a .cpp twin inside
#     the build tree; relative-to-file lookup of `src/...` would land in
#     `<build>/benchmarks/lbm_lid_driven_cavity/src/`, which doesn't
#     exist. Adding the *parent* source dir to the include path makes
#     `#include "src/cell_type.h"` resolve back to the real headers under
#     both backends without rewriting every include in lbm_kernels.h /
#     lbm_lid_driven_cavity.cu.
target_include_directories(carbon_benchmark_lbm_lid_driven_cavity
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
