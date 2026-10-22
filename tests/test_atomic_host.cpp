#include <carbon/detail/atomic.h>

#include <cassert>
#include <cstdio>
#include <thread>
#include <vector>

/**
 * @file
 *
 * Concurrency test for the host branch of @c detail::atomic_add / @c detail::atomic_min.
 *
 * Both primitives carry a device implementation behind @c __CUDA_ARCH__ /
 * @c __HIP_DEVICE_COMPILE__ / @c __SYCL_DEVICE_ONLY__ and a host implementation behind the
 * @c #else . The host branch is easy to mistake for dead code — on the CUDA backend it never
 * runs a kernel — but it is live in two situations: a host TU calling the helper directly,
 * and a SYCL build with an @c omp target, where kernels lower to OpenMP loops so the "device"
 * side of a per-cell accumulate is exactly this code, executing on every core at once.
 *
 * @c atomic_add previously read, added, and stored without synchronization there, which loses
 * updates under contention. This test pins both helpers against that class of regression by
 * hammering one word from many threads and demanding the exact arithmetic result: a lost
 * update shows up as a sum below the expected total, and a lost min as a value above the
 * true minimum.
 *
 * Counts are chosen so the running float sum stays integral and exactly representable
 * (8 threads x 4096 increments of 1.0f peaks at 32768, well inside the 2^24 exact-integer
 * range of binary32), so a mismatch means a lost update rather than rounding.
 */

namespace {

constexpr int kThreads = 8;
constexpr int kIterations = 4096;

void
test_atomic_add_loses_nothing() {
  float counter = 0.0F;

  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([&counter] {
      for (int i = 0; i < kIterations; ++i) {
        (void)carbon::detail::atomic_add(&counter, 1.0F);
      }
    });
  }
  for (std::thread& w : workers) {
    w.join();
  }

  const float expected = static_cast<float>(kThreads) * static_cast<float>(kIterations);
  if (counter != expected) {
    std::fprintf(stderr,
                 "test_atomic_host: atomic_add lost updates: got %.1f, expected %.1f "
                 "(%.0f increments vanished under %d-thread contention)\n",
                 static_cast<double>(counter), static_cast<double>(expected),
                 static_cast<double>(expected - counter), kThreads);
    assert(false && "detail::atomic_add host branch is not atomic");
  }
}

void
test_atomic_min_reaches_true_minimum() {
  // Each thread walks a descending sequence offset by its id, so the threads interleave and
  // the winning value is only correct if every compare-exchange retry is honoured.
  unsigned long long slot = ~0ULL;

  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([&slot, t] {
      for (int i = kIterations; i > 0; --i) {
        const auto candidate = static_cast<unsigned long long>(i * kThreads + t);
        (void)carbon::detail::atomic_min(&slot, candidate);
      }
    });
  }
  for (std::thread& w : workers) {
    w.join();
  }

  // Smallest candidate any thread offers: i == 1, t == 0.
  constexpr auto kExpected = static_cast<unsigned long long>(kThreads);
  if (slot != kExpected) {
    std::fprintf(stderr, "test_atomic_host: atomic_min settled on %llu, expected %llu\n", slot,
                 kExpected);
    assert(false && "detail::atomic_min host branch did not converge to the minimum");
  }
}

}  // namespace

int
main() {
  test_atomic_add_loses_nothing();
  test_atomic_min_reaches_true_minimum();
  std::printf("test_atomic_host: OK (%d threads x %d iterations per helper)\n", kThreads,
              kIterations);
  return 0;
}
