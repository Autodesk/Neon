#include <carbon/process_grid.h>

#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace {

using carbon::DecompositionMode;
using carbon::int32_3d;
using carbon::ProcessGrid;

template <typename Fn>
void
expect_invalid_argument(Fn fn) {
  bool threw = false;
  try {
    fn();
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  assert(threw);
}

template <typename Fn>
void
expect_out_of_range(Fn fn) {
  bool threw = false;
  try {
    fn();
  } catch (const std::out_of_range&) {
    threw = true;
  }
  assert(threw);
}

void
expect_round_trip(const ProcessGrid& grid) {
  for (int32_t rank = 0; rank < grid.world_size(); ++rank) {
    const int32_3d coords = grid.coords_of(rank);
    assert(grid.rank_of(coords) == rank);
  }
}

void
test_slab() {
  const int32_t sizes[]{1, 2, 3, 4, 6, 12};
  for (const int32_t n : sizes) {
    const ProcessGrid grid = ProcessGrid::make(n, DecompositionMode::Slab);
    const int32_3d expected_dims{1, 1, n};
    assert(grid.mode() == DecompositionMode::Slab);
    assert(grid.world_size() == n);
    assert(grid.dims() == expected_dims);
    assert(grid.is_slab());

    for (int32_t rank = 0; rank < n; ++rank) {
      const int32_3d coords = grid.coords_of(rank);
      assert(coords.x == 0);
      assert(coords.y == 0);
      assert(coords.z == rank);
    }
    expect_round_trip(grid);
  }
}

void
test_pencil() {
  const ProcessGrid six = ProcessGrid::make(6, DecompositionMode::Pencil);
  const int32_3d expected_six{1, 2, 3};
  assert(six.mode() == DecompositionMode::Pencil);
  assert(six.dims() == expected_six);
  assert(!six.is_slab());
  expect_round_trip(six);

  const ProcessGrid twelve = ProcessGrid::make(12, DecompositionMode::Pencil);
  const int32_3d expected_twelve{1, 4, 3};
  assert(twelve.world_size() == 12);
  assert(twelve.dims() == expected_twelve);
  assert(twelve.dims().x == 1);
  assert(twelve.dims().y > 1);
  assert(twelve.dims().z > 1);
  expect_round_trip(twelve);
}

void
test_brick() {
  const ProcessGrid eight = ProcessGrid::make(8, DecompositionMode::Brick);
  const int32_3d expected_eight{2, 2, 2};
  assert(eight.mode() == DecompositionMode::Brick);
  assert(eight.dims() == expected_eight);
  assert(!eight.is_slab());
  expect_round_trip(eight);

  const ProcessGrid twelve = ProcessGrid::make(12, DecompositionMode::Brick);
  const int32_3d expected_twelve{2, 2, 3};
  assert(twelve.world_size() == 12);
  assert(twelve.dims() == expected_twelve);
  assert(twelve.dims().x > 1);
  assert(twelve.dims().y > 1);
  assert(twelve.dims().z > 1);
  expect_round_trip(twelve);
}

void
test_custom() {
  const ProcessGrid grid = ProcessGrid::custom(24, 2, 3, 4);
  const int32_3d expected_dims{2, 3, 4};
  assert(grid.mode() == DecompositionMode::Custom);
  assert(grid.world_size() == 24);
  assert(grid.dims() == expected_dims);
  assert(!grid.is_slab());

  const int32_3d origin{0, 0, 0};
  const int32_3d z1{0, 0, 1};
  const int32_3d y1{0, 1, 0};
  const int32_3d x1{1, 0, 0};
  assert(grid.rank_of(origin) == 0);
  assert(grid.rank_of(z1) == 1);
  assert(grid.rank_of(y1) == 4);
  assert(grid.rank_of(x1) == 12);
  expect_round_trip(grid);
}

void
test_invalid() {
  expect_invalid_argument([] { static_cast<void>(ProcessGrid::make(0, DecompositionMode::Slab)); });
  expect_invalid_argument(
      [] { static_cast<void>(ProcessGrid::make(-4, DecompositionMode::Brick)); });
  expect_invalid_argument(
      [] { static_cast<void>(ProcessGrid::make(8, DecompositionMode::Custom)); });

  expect_invalid_argument([] { static_cast<void>(ProcessGrid::custom(0, 1, 1, 1)); });
  expect_invalid_argument([] { static_cast<void>(ProcessGrid::custom(8, 0, 2, 4)); });
  expect_invalid_argument([] { static_cast<void>(ProcessGrid::custom(8, -1, 2, 4)); });
  expect_invalid_argument([] { static_cast<void>(ProcessGrid::custom(8, 2, 2, 3)); });

  const ProcessGrid grid = ProcessGrid::custom(8, 2, 2, 2);
  expect_out_of_range([&] { static_cast<void>(grid.coords_of(-1)); });
  expect_out_of_range([&] { static_cast<void>(grid.coords_of(8)); });

  const int32_3d negative_x{-1, 0, 0};
  const int32_3d past_x{2, 0, 0};
  const int32_3d negative_y{0, -1, 0};
  const int32_3d past_y{0, 2, 0};
  const int32_3d negative_z{0, 0, -1};
  const int32_3d past_z{0, 0, 2};
  expect_out_of_range([&] { static_cast<void>(grid.rank_of(negative_x)); });
  expect_out_of_range([&] { static_cast<void>(grid.rank_of(past_x)); });
  expect_out_of_range([&] { static_cast<void>(grid.rank_of(negative_y)); });
  expect_out_of_range([&] { static_cast<void>(grid.rank_of(past_y)); });
  expect_out_of_range([&] { static_cast<void>(grid.rank_of(negative_z)); });
  expect_out_of_range([&] { static_cast<void>(grid.rank_of(past_z)); });
}

}  // namespace

int main() {
  test_slab();
  test_pencil();
  test_brick();
  test_custom();
  test_invalid();
}