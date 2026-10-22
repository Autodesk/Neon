#include <carbon/carbon.h>
#include <carbon/cpu.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using carbon::detail::Allocation;
using carbon::detail::AllocationLocation;
using carbon::detail::BufferGroup;

// Build a single-entry host carbon::Buffer<T> with @p n elements.
template <typename T>
auto
make_host_group_1(size_t n) -> carbon::Buffer<T> {
  std::vector<Allocation<T>> v;
  v.push_back(carbon::Cpu::buffer<T>(n));
  return BufferGroup<T>::from_vector(std::move(v));
}

// Build a two-entry host carbon::Buffer<T> with sizes (n0, n1).
template <typename T>
auto
make_host_group_2(size_t n0,
                  size_t n1) -> carbon::Buffer<T> {
  std::vector<Allocation<T>> v;
  v.push_back(carbon::Cpu::buffer<T>(n0));
  v.push_back(carbon::Cpu::buffer<T>(n1));
  return BufferGroup<T>::from_vector(std::move(v));
}

// Buffer::unmanaged wraps a raw pointer without taking ownership; owns_storage reflects that.
auto
test_unmanaged_and_owns_storage() -> void {
  auto owned = carbon::Cpu::buffer<int>(8U);
  assert(owned);
  assert(owned.owns_storage());
  for (size_t i = 0; i < 8U; ++i) {
    owned.data()[i] = static_cast<int>(i);
  }
  int stack_storage[4] = {10, 11, 12, 13};
  auto wrap = Allocation<int>::unmanaged(AllocationLocation::Cpu, stack_storage, 4U, -1, -1);
  assert(wrap);
  assert(!wrap.owns_storage());
  assert(wrap.data() == stack_storage);
  assert(wrap.size() == 4U);
  assert(wrap.size_bytes() == 4U * sizeof(int));
  {
    auto wrap2 = wrap;
    assert(wrap2.data() == stack_storage);
  }
  assert(stack_storage[0] == 10);
}

// as_byte_view shares the control block of its source and exposes the right byte count.
auto
test_as_byte_view() -> void {
  auto owned = carbon::Cpu::buffer<float>(5U);
  for (size_t i = 0; i < 5U; ++i) {
    owned.data()[i] = 1.0F + static_cast<float>(i);
  }
  auto bytes = owned.as_byte_view();
  assert(bytes);
  assert(bytes.size() == 5U * sizeof(float));
  assert(bytes.owns_storage());
  assert(bytes.location() == owned.location());
  assert(static_cast<const void*>(bytes.data()) == static_cast<const void*>(owned.data()));

  float stack[3] = {0.0F, 0.0F, 0.0F};
  auto raw = Allocation<float>::unmanaged(AllocationLocation::Cpu, stack, 3U, -1, -1);
  auto raw_bytes = raw.as_byte_view();
  assert(!raw_bytes.owns_storage());
  assert(raw_bytes.size() == 3U * sizeof(float));
}

// Full-buffer Transfer over a single-entry host carbon::Buffer.
auto
test_full_buffer_single_entry() -> void {
  constexpr size_t n = 16U;
  auto src = make_host_group_1<float>(n);
  auto dst = make_host_group_1<float>(n);
  for (size_t i = 0; i < n; ++i) {
    src[0].data()[i] = 1.0F + static_cast<float>(i);
  }
  std::memset(dst[0].data(), 0, dst[0].size_bytes());

  carbon::System system((std::vector<carbon::detail::Gpu>{}));
  carbon::Transfer<float> t(system, src, dst);
  assert(t.src().size() == 1U);
  assert(t.src_first_element().size() == 1U && t.src_first_element()[0] == 0U);
  assert(t.src_counts().size() == 1U && t.src_counts()[0] == n);
  assert(t.dst_first_element()[0] == 0U);
  assert(t.dst_counts()[0] == n);

  carbon::Stream stream;
  t(stream);

  for (size_t i = 0; i < n; ++i) {
    assert(dst[0].data()[i] == src[0].data()[i]);
  }
}

// Full-buffer Transfer over a two-entry host carbon::Buffer (different per-GPU sizes).
auto
test_full_buffer_two_entries() -> void {
  constexpr size_t n0 = 4U;
  constexpr size_t n1 = 7U;
  auto src = make_host_group_2<int>(n0, n1);
  auto dst = make_host_group_2<int>(n0, n1);
  for (size_t i = 0; i < n0; ++i) {
    src[0].data()[i] = 100 + static_cast<int>(i);
    dst[0].data()[i] = -1;
  }
  for (size_t i = 0; i < n1; ++i) {
    src[1].data()[i] = 200 + static_cast<int>(i);
    dst[1].data()[i] = -1;
  }

  carbon::System system((std::vector<carbon::detail::Gpu>{}));
  carbon::Transfer<int> t(system, src, dst);
  carbon::Stream stream;
  t(stream);

  for (size_t i = 0; i < n0; ++i) {
    assert(dst[0].data()[i] == src[0].data()[i]);
  }
  for (size_t i = 0; i < n1; ++i) {
    assert(dst[1].data()[i] == src[1].data()[i]);
  }
}

// Sub-range Transfer: copy a window inside each per-GPU buffer using element offsets.
auto
test_subrange_host_transfer() -> void {
  constexpr size_t n = 10U;
  auto src = make_host_group_1<float>(n);
  auto dst = make_host_group_1<float>(n);
  for (size_t i = 0; i < n; ++i) {
    src[0].data()[i] = 1000.0F + static_cast<float>(i);
    dst[0].data()[i] = -1.0F;
  }
  constexpr size_t src_start = 2U;
  constexpr size_t dst_start = 5U;
  constexpr size_t len = 3U;

  std::vector<size_t> sf{src_start};
  std::vector<size_t> df{dst_start};
  std::vector<size_t> sc{len};
  std::vector<size_t> dc{len};
  carbon::System system((std::vector<carbon::detail::Gpu>{}));
  carbon::Transfer<float> t(system, src, std::move(sf), std::move(sc), dst, std::move(df),
                            std::move(dc));

  carbon::Stream stream;
  t(stream);

  for (size_t i = 0; i < n; ++i) {
    if (i >= dst_start && i < dst_start + len) {
      assert(dst[0].data()[i] == src[0].data()[i - dst_start + src_start]);
    } else {
      assert(dst[0].data()[i] == -1.0F);
    }
  }
}

// Per-GPU zero-count entries are silently skipped; non-zero entries still copy.
auto
test_zero_count_entry_skipped() -> void {
  constexpr size_t n0 = 4U;
  constexpr size_t n1 = 5U;
  auto src = make_host_group_2<int>(n0, n1);
  auto dst = make_host_group_2<int>(n0, n1);
  for (size_t i = 0; i < n0; ++i) {
    src[0].data()[i] = 11 + static_cast<int>(i);
    dst[0].data()[i] = 0;
  }
  for (size_t i = 0; i < n1; ++i) {
    src[1].data()[i] = 99;
    dst[1].data()[i] = 7;
  }
  std::vector<size_t> sf{0U, 0U};
  std::vector<size_t> df{0U, 0U};
  std::vector<size_t> sc{n0, 0U};
  std::vector<size_t> dc{n0, 0U};
  carbon::System system((std::vector<carbon::detail::Gpu>{}));
  carbon::Transfer<int> t(system, src, std::move(sf), std::move(sc), dst, std::move(df),
                          std::move(dc));

  carbon::Stream stream;
  t(stream);

  for (size_t i = 0; i < n0; ++i) {
    assert(dst[0].data()[i] == src[0].data()[i]);
  }
  for (size_t i = 0; i < n1; ++i) {
    assert(dst[1].data()[i] == 7);
  }
}

// Validation: buffer cardinality mismatch, byte-size disagreement, and out-of-range raw pointer.
auto
test_validation_errors() -> void {
  carbon::System system((std::vector<carbon::detail::Gpu>{}));
  bool threw = false;
  {
    auto src = make_host_group_1<float>(4U);
    auto dst = make_host_group_2<float>(4U, 4U);
    try {
      carbon::Transfer<float> t(system, src, dst);
      (void)t;
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    assert(threw);
  }

  threw = false;
  {
    auto src = make_host_group_1<float>(4U);
    auto dst = make_host_group_1<float>(8U);
    try {
      carbon::Transfer<float> t(system, src, dst);
      (void)t;
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    assert(threw);
  }

  threw = false;
  {
    auto src = make_host_group_1<float>(4U);
    auto dst = make_host_group_1<float>(4U);
    std::vector<size_t> sf{3U};  // 3 + 2 = 5 > size 4
    std::vector<size_t> df{0U};
    std::vector<size_t> sc{2U};
    std::vector<size_t> dc{2U};
    try {
      carbon::Transfer<float> t(system, src, std::move(sf), std::move(sc), dst, std::move(df),
                                std::move(dc));
      (void)t;
    } catch (const std::out_of_range&) {
      threw = true;
    }
    assert(threw);
  }

  threw = false;
  {
    auto src = make_host_group_1<float>(8U);
    auto dst = make_host_group_1<float>(8U);
    std::vector<size_t> sf{0U};
    std::vector<size_t> df{0U};
    std::vector<size_t> sc{8U};
    std::vector<size_t> dc{4U};
    try {
      carbon::Transfer<float> t(system, src, std::move(sf), std::move(sc), dst, std::move(df),
                                std::move(dc));
      (void)t;
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    assert(threw);
  }
}

}  // namespace

auto
main() -> int {
  carbon::init();
  test_unmanaged_and_owns_storage();
  test_as_byte_view();
  test_full_buffer_single_entry();
  test_full_buffer_two_entries();
  test_subrange_host_transfer();
  test_zero_count_entry_skipped();
  test_validation_errors();
  return 0;
}
