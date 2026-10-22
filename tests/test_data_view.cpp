#include <carbon/carbon.h>
#include <carbon/data_view.h>

#include <cassert>
#include <sstream>
#include <stdexcept>
#include <string>

int
main() {
  using carbon::DataView;

  static_assert(carbon::data_view_count == 3);
  static_assert(static_cast<int>(DataView::Standard) == 0);
  static_assert(static_cast<int>(DataView::Internal) == 1);
  static_assert(static_cast<int>(DataView::Boundary) == 2);

  assert(carbon::to_string(DataView::Standard) == "standard");
  assert(carbon::to_string(DataView::Internal) == "internal");
  assert(carbon::to_string(DataView::Boundary) == "boundary");

  assert(carbon::to_int(DataView::Standard) == 0);
  assert(carbon::to_int(DataView::Internal) == 1);
  assert(carbon::to_int(DataView::Boundary) == 2);

  for (const DataView v : carbon::data_view_options()) {
    assert(carbon::from_int(carbon::to_int(v)) == v);
    assert(carbon::from_string(carbon::to_string(v)) == v);
  }

  assert(carbon::from_string("standard") == DataView::Standard);
  assert(carbon::from_string("Standard") == DataView::Standard);
  assert(carbon::from_string("STANDARD") == DataView::Standard);
  assert(carbon::from_string("sTaNdArD") == DataView::Standard);
  assert(carbon::from_string("INTERNAL") == DataView::Internal);
  assert(carbon::from_string("Boundary") == DataView::Boundary);

  const auto options = carbon::data_view_options();
  assert(options.size() == 3);
  assert(options[0] == DataView::Standard);
  assert(options[1] == DataView::Internal);
  assert(options[2] == DataView::Boundary);

  {
    std::ostringstream os;
    os << DataView::Standard << ',' << DataView::Internal << ',' << DataView::Boundary;
    assert(os.str() == "standard,internal,boundary");
  }

  bool threw = false;
  try {
    (void)carbon::from_int(-1);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    (void)carbon::from_int(3);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    (void)carbon::from_int(static_cast<int>(carbon::data_view_count));
  } catch (const std::out_of_range&) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    (void)carbon::from_string("weird");
  } catch (const std::out_of_range&) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    (void)carbon::from_string("");
  } catch (const std::out_of_range&) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    (void)carbon::from_string("  standard  ");
  } catch (const std::out_of_range&) {
    threw = true;
  }
  assert(threw);

  return 0;
}
