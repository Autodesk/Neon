#pragma once

/**
 * Per-parent-family hooks used to build a @c carbon::SubGrid : how to enumerate the parent's
 * cells of a @ref DataView on the host, where a cell's selection flag lives in the downloaded
 * flag buffer, and which parent span a host kernel may dereference.
 *
 * Supported parents: @ref carbon::Dense and @ref carbon::Block . The primary template is left
 * undefined so any other parent fails at @c SubGrid instantiation.
 */

#include <carbon/block.h>
#include <carbon/data_view.h>
#include <carbon/dense.h>
#include <carbon/integer3d.h>

#include <cstddef>
#include <cstdint>

namespace carbon::detail::sub_grid {

template <typename G>
struct ParentTraits;

template <typename G>
inline constexpr bool has_parent_traits_v = requires {
  sizeof(ParentTraits<G>);
};

template <>
struct ParentTraits<::carbon::Dense> {
  using Span = ::carbon::Dense::Span;
  using Idx = ::carbon::Dense::Idx;

  /** Dense spans are pure geometry (no pointers), so the device span serves the host too. */
  [[nodiscard]] static auto
  host_span(const ::carbon::Dense& grid,
            size_t gpu) -> Span {
    return grid.spans()[gpu];
  }

  /** Calls @p fn for every cell of @p view in the parent's iteration order. */
  template <typename F>
  static auto
  for_each_cell(const Span& span,
                ::carbon::DataView view,
                F&& fn) -> void {
    const int64_3d ext = span.iter_extent(view);
    for (int64_t z = 0; z < ext.z; ++z) {
      for (int64_t y = 0; y < ext.y; ++y) {
        for (int64_t x = 0; x < ext.x; ++x) {
          const auto r = span.set_and_validate(view, x, y, z);
          if (r.valid) {
            fn(r.idx);
          }
        }
      }
    }
  }

  /** Offset of @p idx 's flag in a host copy of a one-component parent field. */
  [[nodiscard]] static auto
  flag_offset(const Span& span,
              const Idx& idx) -> int64_t {
    return span.linear_offset(idx);
  }
};

template <typename SBlock>
struct ParentTraits<::carbon::Block<SBlock>> {
  using Grid = ::carbon::Block<SBlock>;
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;

  /**
   * A span over @p grid 's own host topology tables (bitmasks, connectivity, origins).
   *
   * Built here instead of taken from @c Block::host_spans() , whose memoised spans are copied
   * along with the @c Block and are not thread-safe. The host tables cover owned bricks only,
   * so a host-side @c neighbour that crosses the slab cut (into a ghost slot) is out of range,
   * exactly as for @c Block 's own host kernels.
   */
  [[nodiscard]] static auto
  host_span(const Grid& grid,
            size_t gpu) -> Span {
    const uint32_t n_blocks = grid.block_count(gpu);
    if (n_blocks == 0U) {
      return Span{};
    }
    return Span(grid.host_bitmasks(gpu).data(), n_blocks, grid.n_internal_blocks(gpu),
                grid.host_connectivity(gpu).data(), grid.host_origins(gpu).data());
  }

  /**
   * Calls @p fn for every active voxel of @p view : bricks in storage order (Internal bricks,
   * then Boundary bricks), voxels z-y-x within a brick, which is also their storage order.
   * @p span must be a host span (the bitmask is dereferenced).
   */
  template <typename F>
  static auto
  for_each_cell(const Span& span,
                ::carbon::DataView view,
                F&& fn) -> void {
    const uint32_t n = span.iter_block_count(view);
    for (uint32_t it = 0; it < n; ++it) {
      const uint32_t b = span.view_block_idx(view, it);
      for (int iz = 0; iz < Span::MemZ; ++iz) {
        for (int iy = 0; iy < Span::MemY; ++iy) {
          for (int ix = 0; ix < Span::MemX; ++ix) {
            const auto r = span.set_and_validate(b, ix, iy, iz);
            if (r.valid) {
              fn(r.idx);
            }
          }
        }
      }
    }
  }

  /** Offset of @p idx 's flag in the owned-brick buffer returned by @c Block::download_to_host . */
  [[nodiscard]] static auto
  flag_offset(const Span& span,
              const Idx& idx) -> int64_t {
    return span.linear_offset(idx, 1, 0);
  }
};

}  // namespace carbon::detail::sub_grid
