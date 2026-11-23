// SubGrid compile-time rules. Each CARBON_CF_CASE is built by its own ctest entry (see
// tests/CMakeLists.txt); case 0 must compile, every other case must fail with the
// diagnostic the test registration matches.
//
//   1  read_stencil on a SubGrid field (S has no halo)
//   2  neighbour access through a SubGrid partition
//   3  writing a SubGrid partition with the parent's Span / Idx
//   4  a SubGrid of a SubGrid
//   5  Kernel::compute_universal over a SubGrid
//   6  read_stencil on a field of a SubGrid over a Block
//   7  a SubGrid over an unsupported parent grid

#include <carbon/carbon.h>

#ifndef CARBON_CF_CASE
#error "define CARBON_CF_CASE"
#endif

using S = carbon::SubGrid<carbon::Dense>;

[[maybe_unused]] auto
build(const S& sub,
      S::Field<float,
               1>& f) -> carbon::Kernel {
  return sub.make_kernel([f](carbon::KernelLoader& L) {
#if CARBON_CF_CASE == 1
    auto p = L.read_stencil(f);
#else
    auto p = L.read_write(f);
#endif
    return [=] CARBON_HD(const S::Span& s, const S::Idx& i) {
#if CARBON_CF_CASE == 2
      (void)p.neighbour(s, i, carbon::int8_3d{0, 0, 1}, 0);
#elif CARBON_CF_CASE == 3
      const carbon::Dense::Span& ps = s;
      const carbon::Dense::Idx& pi = i;
      p.write(ps, pi, 0, 1.0F);
#else
      (void)p.read(s, i, 0);
#endif
    };
  });
}

using SB = carbon::SubGrid<carbon::Block<>>;

#if CARBON_CF_CASE == 0 || CARBON_CF_CASE == 6
[[maybe_unused]] auto
build_block(const SB& sub,
            SB::Field<float,
                      1>& f) -> carbon::Kernel {
  return sub.make_kernel([f](carbon::KernelLoader& L) {
#if CARBON_CF_CASE == 6
    auto p = L.read_stencil(f);
#else
    auto p = L.read_write(f);
#endif
    return [=] CARBON_HD(const SB::Span& s, const SB::Idx& i) {
      p.write(s, i, 0, p.read(s, i, 0) + 1.0F);
    };
  });
}
#endif

#if CARBON_CF_CASE == 7
[[maybe_unused]] auto unsupported_size = sizeof(carbon::SubGrid<carbon::DenseDisg>);
#endif

#if CARBON_CF_CASE == 4
[[maybe_unused]] carbon::SubGrid<S>* nested = nullptr;
[[maybe_unused]] auto nested_size = sizeof(carbon::SubGrid<S>);
#endif

#if CARBON_CF_CASE == 5
[[maybe_unused]] auto
build_universal(const S& sub,
                S::Field<float,
                         1>& f) -> carbon::Kernel {
  return carbon::Kernel::compute_universal(sub, [f](carbon::KernelLoader& L) {
    auto p = L.write(f);
    return [=] CARBON_HD(const S::Span& s, const S::Idx& i) {
      p.write(s, i, 0, 1.0F);
    };
  });
}
#endif

int
main() {
  return 0;
}
