// detail/qcriterion.inl — Q-criterion kernel for <carbon/qcriterion.h>.

#include <carbon/integer3d.h>

namespace carbon {
namespace detail {

template <typename T>
[[nodiscard]] CARBON_HD constexpr auto
frobenius_norm_sq(const T (&m)[3][3]) noexcept -> T {
  T sum = T{0};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      sum += m[i][j] * m[i][j];
    }
  }
  return sum;
}

}  // namespace detail

template <typename Grid,
          typename T,
          int VelocityComponents>
inline auto
make_qcriterion_kernel(const Grid& grid,
                       typename Grid::template Field<T,
                                                     VelocityComponents>& velocity,
                       typename Grid::template Field<T,
                                                     1>& q_out,
                       T velocity_scale) -> Kernel {
  static_assert(VelocityComponents == 3,
                "make_qcriterion_kernel: velocity field must have exactly 3 components");

  return grid.make_kernel([&velocity, &q_out, velocity_scale](KernelLoader& loader) {
    const auto u = loader.read_stencil(velocity);
    auto out = loader.write(q_out);
    return [u, out, velocity_scale]
           CARBON_HD(typename Grid::Span span, typename Grid::Idx idx) -> void {
             const T inv_2h = velocity_scale / (T{2} * static_cast<T>(span.spacing()));

             const int8_3d xp{1, 0, 0};
             const int8_3d xm{-1, 0, 0};
             const int8_3d yp{0, 1, 0};
             const int8_3d ym{0, -1, 0};
             const int8_3d zp{0, 0, 1};
             const int8_3d zm{0, 0, -1};
             const int8_3d offs[3][2] = {{xp, xm}, {yp, ym}, {zp, zm}};

             T j[3][3]{};
             for (int spatial = 0; spatial < 3; ++spatial) {
               for (int comp = 0; comp < 3; ++comp) {
                 const auto n_p = u.neighbour(span, idx, offs[spatial][0], comp);
                 const auto n_m = u.neighbour(span, idx, offs[spatial][1], comp);
                 if (!n_p.valid || !n_m.valid) {
                   out.write(span, idx, 0, T{0});
                   return;
                 }
                 j[comp][spatial] = (n_p.value - n_m.value) * inv_2h;
               }
             }

             T s[3][3]{};
             T omega[3][3]{};
             for (int i = 0; i < 3; ++i) {
               for (int j_idx = 0; j_idx < 3; ++j_idx) {
                 s[i][j_idx] = T{0.5} * (j[i][j_idx] + j[j_idx][i]);
                 omega[i][j_idx] = T{0.5} * (j[i][j_idx] - j[j_idx][i]);
               }
             }

             const T q = T{0.5} * (detail::frobenius_norm_sq(omega) - detail::frobenius_norm_sq(s));
             out.write(span, idx, 0, q);
           };
  });
}

}  // namespace carbon
