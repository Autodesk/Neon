#pragma once

#include <cassert>
#include <exception>

/**
 * @file
 *
 * Lightweight CARBON_EXPECT_THROW / CARBON_EXPECT_NOTHROW macros for the unit-test
 * suite. carbon does not pull in GoogleTest, so the dozens of try/catch sites in
 * tests/ either roll their own bool flag pattern or skip the assertion altogether.
 *
 * Usage:
 *
 *   CARBON_EXPECT_THROW(grid.newField<int, 1>(MemorySide::Device), std::invalid_argument);
 *
 * The macro evaluates @p expr inside a try/catch that:
 *   - asserts the listed @p exc_type is caught (assertion message points at the
 *     macro call site so failures are obvious),
 *   - asserts no other exception type slipped through,
 *   - asserts the expression did throw at all.
 *
 * Two assertion failures are possible:
 *   1. "expected EXC, but expression did not throw" -- expr returned normally.
 *   2. "expected EXC, but a different exception type was thrown" -- expr threw
 *      something else.
 *
 * Designed to be drop-in for the existing
 *   bool threw = false;
 *   try { ...; } catch (const std::invalid_argument&) { threw = true; }
 *   assert(threw);
 * pattern.
 */

#define CARBON_EXPECT_THROW(expr, exc_type)                                        \
  do {                                                                             \
    bool carbon_test_threw_expected_ = false;                                      \
    try {                                                                          \
      (void)(expr);                                                                \
    } catch (const exc_type&) {                                                    \
      carbon_test_threw_expected_ = true;                                          \
    } catch (...) {                                                                \
      assert(false && #expr ": expected " #exc_type                                \
                            ", but a different exception type was thrown");        \
    }                                                                              \
    assert(carbon_test_threw_expected_ && #expr ": expected " #exc_type            \
                                                ", but expression did not throw"); \
  } while (false)

#define CARBON_EXPECT_NOTHROW(expr)                                           \
  do {                                                                        \
    try {                                                                     \
      (void)(expr);                                                           \
    } catch (...) {                                                           \
      assert(false && #expr ": expected no exception, but expression threw"); \
    }                                                                         \
  } while (false)
