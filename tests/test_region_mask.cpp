#include <carbon/carbon.h>
#include <carbon/region_mask.h>

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

template <typename F>
auto
throws_out_of_range(F&& f) -> bool {
  try {
    f();
  } catch (const std::out_of_range&) {
    return true;
  }
  return false;
}

}  // namespace

int
main() {
  using carbon::RegionMask;

  // Compile-time usability: the masks a kernel declares are typically constants.
  static_assert(RegionMask{}.empty());
  static_assert(RegionMask::all().is_all());
  static_assert(RegionMask::all().count() == 256U);
  static_assert(RegionMask{0, 3, 7}.count() == 3U);
  static_assert(!RegionMask::bit(1).intersects(RegionMask::bit(2)));
  static_assert(RegionMask::range(0, 128) == ~RegionMask::range(128, 256));
  static_assert(RegionMask::from_label(200) == RegionMask::bit(200));

  {
    const RegionMask empty;
    assert(empty.empty());
    assert(!empty.is_all());
    assert(empty == RegionMask::none());
    assert(empty.count() == 0U);
    assert(!empty.intersects(RegionMask::all()));
    assert(!empty.intersects(empty));
  }

  {
    const RegionMask all = RegionMask::all();
    for (unsigned b = 0; b < RegionMask::kBits; ++b) {
      assert(all.test(b));
      assert(all.intersects(RegionMask::bit(b)));
    }
    assert(all.intersects(all));
    assert(!all.test(256U));
    assert((~all).empty());
  }

  // Word boundaries: bits 63/64 live in different words, 255 is the last bit.
  for (const unsigned b : {0U, 1U, 63U, 64U, 127U, 128U, 191U, 192U, 255U}) {
    const RegionMask m = RegionMask::bit(b);
    assert(m.count() == 1U);
    assert(m.test(b));
    assert(m.word(b / 64U) == (std::uint64_t{1} << (b % 64U)));
    assert(m == RegionMask::from_label(static_cast<std::uint8_t>(b)));
    assert(m.test(static_cast<std::uint8_t>(b)));
    for (const unsigned other : {0U, 63U, 64U, 255U}) {
      assert(m.intersects(RegionMask::bit(other)) == (b == other));
    }
  }

  {
    assert(RegionMask::range(5, 5).empty());
    assert(RegionMask::range(0, 256).is_all());
    const RegionMask r = RegionMask::range(60, 70);
    assert(r.count() == 10U);
    assert(!r.test(59U) && r.test(60U) && r.test(63U) && r.test(64U) && r.test(69U));
    assert(!r.test(70U));
    assert(r.intersects(RegionMask::bit(64)));
    assert(!r.intersects(RegionMask::range(70, 256)));
  }

  // The full-partition idiom: a region and its complement split every bit.
  {
    const RegionMask bc = RegionMask{0, 17, 200};
    const RegionMask rest = ~bc;
    assert(!bc.intersects(rest));
    assert((bc | rest).is_all());
    assert((bc & rest).empty());
    assert((bc ^ RegionMask::all()) == rest);
    assert(rest.count() == 253U);
  }

  {
    RegionMask m;
    m.set(4).set(130);
    assert(m.test(4U) && m.test(130U) && m.count() == 2U);
    m.reset(4);
    assert(!m.test(4U) && m == RegionMask::bit(130));
    RegionMask n = RegionMask::bit(1);
    n |= RegionMask::bit(2);
    n &= RegionMask{2, 3};
    assert(n == RegionMask::bit(2));
    n ^= RegionMask::bit(2);
    assert(n.empty());
    assert(n != RegionMask::bit(2));
  }

  assert(throws_out_of_range([] {
    (void)RegionMask::bit(256);
  }));
  assert(throws_out_of_range([] {
    (void)RegionMask{1, 300};
  }));
  assert(throws_out_of_range([] {
    (void)RegionMask::range(10, 9);
  }));
  assert(throws_out_of_range([] {
    (void)RegionMask::range(0, 257);
  }));
  assert(throws_out_of_range([] {
    RegionMask m;
    m.set(256);
  }));
  assert(throws_out_of_range([] {
    RegionMask m;
    m.reset(1000);
  }));

  assert(carbon::to_string(RegionMask::all()) == "all");
  assert(carbon::to_string(RegionMask{}) == "none");
  assert(carbon::to_string(RegionMask{3}) == "{3}");
  assert(carbon::to_string(RegionMask{0, 3, 5, 6, 7, 8, 9}) == "{0,3,5-9}");
  assert(carbon::to_string(RegionMask::range(62, 66) | RegionMask::bit(255)) == "{62-65,255}");
  assert(carbon::to_string(~RegionMask::bit(0)) == "{1-255}");

  return 0;
}
