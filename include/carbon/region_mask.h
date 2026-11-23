#pragma once

/**
 * @file region_mask.h
 * @brief @c carbon::RegionMask — a 256-bit, user-defined sub-domain label set that refines
 *        @ref AccessToken dependencies below whole-field granularity.
 *
 * The user owns the meaning of every bit: bit @c b names a cell set @c R(b) of the domain
 * (a set of boundary-condition voxels, the lower half of the domain, ...). Carbon never
 * learns the geometry; it only relies on the contract below to decide that two accesses to
 * the same field cannot touch the same cells, so the @ref Skeleton may run them concurrently.
 *
 * @par Contract
 * - **Bits are atoms.** The sets @c R(b) must be pairwise disjoint, and @c R(M) is the union
 *   of @c R(b) over the bits set in @c M . Under that rule, masks that do not intersect name
 *   disjoint cell sets.
 * - **All ones is the whole field**, including any cell no bit covers. It is the default
 *   footprint of every access, so a pipeline that never names a mask behaves exactly as
 *   before.
 * - Recommended: make the bits a full partition (every cell in exactly one bit, e.g. bit 0
 *   for "everything else"), so @c ~M really means "the rest of the domain". 256 bits is the
 *   range of a @c uint8_t , so a per-cell @c uint8_t label field filtered with
 *   @ref test(std::uint8_t) gives that partition for free.
 * - Masks are a declaration, not a launch restriction: a kernel declared on @c M must only
 *   touch cells in @c R(M) , and must filter itself.
 *
 * Default construction yields the empty mask (like @c std::bitset ); use @ref all for the
 * whole field.
 */

#include <carbon/cuda_annotation.h>
#include <carbon/export.h>

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace carbon {

class RegionMask {
public:
  static constexpr unsigned kBits = 256U;
  static constexpr unsigned kWords = kBits / 64U;

  /** Empty mask (no bit set). */
  constexpr RegionMask() noexcept = default;

  /**
   * @brief Mask with exactly the listed bits set, e.g. @c RegionMask{0, 3, 7} .
   * @throws std::out_of_range if any bit is @c >= kBits .
   */
  constexpr RegionMask(std::initializer_list<unsigned> bits) {
    for (const unsigned b : bits) {
      set(b);
    }
  }

  [[nodiscard]] CARBON_HD static constexpr auto
  all() noexcept -> RegionMask {
    RegionMask m;
    for (unsigned i = 0; i < kWords; ++i) {
      m.w_[i] = ~std::uint64_t{0};
    }
    return m;
  }

  [[nodiscard]] CARBON_HD static constexpr auto
  none() noexcept -> RegionMask {
    return RegionMask{};
  }

  /**
   * @brief Single-bit mask.
   * @throws std::out_of_range if @p b is @c >= kBits .
   */
  [[nodiscard]] static constexpr auto
  bit(unsigned b) -> RegionMask {
    RegionMask m;
    m.set(b);
    return m;
  }

  /**
   * @brief Bits in the half-open range @c [lo, hi) . An empty range yields the empty mask.
   * @throws std::out_of_range if @p lo > @p hi or @p hi > kBits .
   */
  [[nodiscard]] static constexpr auto
  range(unsigned lo,
        unsigned hi) -> RegionMask {
    if (lo > hi || hi > kBits) {
      throw std::out_of_range("RegionMask::range: requires lo <= hi <= 256");
    }
    RegionMask m;
    for (unsigned b = lo; b < hi; ++b) {
      m.w_[b / 64U] |= std::uint64_t{1} << (b % 64U);
    }
    return m;
  }

  /** Single-bit mask for a per-cell @c uint8_t label; always in range. */
  [[nodiscard]] CARBON_HD static constexpr auto
  from_label(std::uint8_t label) noexcept -> RegionMask {
    RegionMask m;
    m.w_[label / 64U] = std::uint64_t{1} << (label % 64U);
    return m;
  }

  /**
   * @brief Set bit @p b .
   * @throws std::out_of_range if @p b is @c >= kBits .
   */
  constexpr auto
  set(unsigned b) -> RegionMask& {
    check_bit_(b);
    w_[b / 64U] |= std::uint64_t{1} << (b % 64U);
    return *this;
  }

  /**
   * @brief Clear bit @p b .
   * @throws std::out_of_range if @p b is @c >= kBits .
   */
  constexpr auto
  reset(unsigned b) -> RegionMask& {
    check_bit_(b);
    w_[b / 64U] &= ~(std::uint64_t{1} << (b % 64U));
    return *this;
  }

  /** Whether bit @p b is set; @c false for any @p b @c >= kBits . */
  [[nodiscard]] CARBON_HD constexpr auto
  test(unsigned b) const noexcept -> bool {
    return b < kBits && ((w_[b / 64U] >> (b % 64U)) & std::uint64_t{1}) != 0U;
  }

  /** Whether the bit named by a per-cell @c uint8_t label is set. */
  [[nodiscard]] CARBON_HD constexpr auto
  test(std::uint8_t label) const noexcept -> bool {
    return ((w_[label / 64U] >> (label % 64U)) & std::uint64_t{1}) != 0U;
  }

  /** Whether the two masks share at least one bit — i.e. may name overlapping cells. */
  [[nodiscard]] CARBON_HD constexpr auto
  intersects(const RegionMask& other) const noexcept -> bool {
    for (unsigned i = 0; i < kWords; ++i) {
      if ((w_[i] & other.w_[i]) != 0U) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  empty() const noexcept -> bool {
    for (unsigned i = 0; i < kWords; ++i) {
      if (w_[i] != 0U) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  is_all() const noexcept -> bool {
    for (unsigned i = 0; i < kWords; ++i) {
      if (w_[i] != ~std::uint64_t{0}) {
        return false;
      }
    }
    return true;
  }

  /** Number of bits set. */
  [[nodiscard]] CARBON_HD constexpr auto
  count() const noexcept -> unsigned {
    unsigned n = 0;
    for (unsigned i = 0; i < kWords; ++i) {
      for (std::uint64_t v = w_[i]; v != 0U; v &= v - 1U) {
        ++n;
      }
    }
    return n;
  }

  /** Raw 64-bit word @p i (bit @c b lives in word @c b/64 at position @c b%64 ). */
  [[nodiscard]] CARBON_HD constexpr auto
  word(unsigned i) const noexcept -> std::uint64_t {
    return i < kWords ? w_[i] : 0U;
  }

  CARBON_HD constexpr auto
  operator&=(const RegionMask& o) noexcept -> RegionMask& {
    for (unsigned i = 0; i < kWords; ++i) {
      w_[i] &= o.w_[i];
    }
    return *this;
  }

  CARBON_HD constexpr auto
  operator|=(const RegionMask& o) noexcept -> RegionMask& {
    for (unsigned i = 0; i < kWords; ++i) {
      w_[i] |= o.w_[i];
    }
    return *this;
  }

  CARBON_HD constexpr auto
  operator^=(const RegionMask& o) noexcept -> RegionMask& {
    for (unsigned i = 0; i < kWords; ++i) {
      w_[i] ^= o.w_[i];
    }
    return *this;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  operator~() const noexcept -> RegionMask {
    RegionMask m;
    for (unsigned i = 0; i < kWords; ++i) {
      m.w_[i] = ~w_[i];
    }
    return m;
  }

  [[nodiscard]] CARBON_HD friend constexpr auto
  operator&(RegionMask a,
            const RegionMask& b) noexcept -> RegionMask {
    return a &= b;
  }

  [[nodiscard]] CARBON_HD friend constexpr auto
  operator|(RegionMask a,
            const RegionMask& b) noexcept -> RegionMask {
    return a |= b;
  }

  [[nodiscard]] CARBON_HD friend constexpr auto
  operator^(RegionMask a,
            const RegionMask& b) noexcept -> RegionMask {
    return a ^= b;
  }

  [[nodiscard]] CARBON_HD friend constexpr auto
  operator==(const RegionMask& a,
             const RegionMask& b) noexcept -> bool {
    for (unsigned i = 0; i < kWords; ++i) {
      if (a.w_[i] != b.w_[i]) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] CARBON_HD friend constexpr auto
  operator!=(const RegionMask& a,
             const RegionMask& b) noexcept -> bool {
    return !(a == b);
  }

private:
  static constexpr auto
  check_bit_(unsigned b) -> void {
    if (b >= kBits) {
      throw std::out_of_range("RegionMask: bit index must be < 256");
    }
  }

  // Plain array rather than std::array: the type is captured by value into device lambdas
  // and must stay usable from CUDA / HIP / SYCL device code without relaxed-constexpr.
  std::uint64_t w_[kWords]{};
};

static_assert(std::is_trivially_copyable_v<RegionMask>);
static_assert(sizeof(RegionMask) == 32U);

/**
 * @brief Human-readable form used by diagnostics and the DOT exporters: @c "all" ,
 *        @c "none" , or a compressed bit list such as @c "{0,3,5-9}" .
 */
[[nodiscard]] CARBON_API auto
to_string(const RegionMask& mask) -> std::string;

}  // namespace carbon
