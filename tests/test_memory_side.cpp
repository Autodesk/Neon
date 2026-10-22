#include <carbon/carbon.h>
#include <carbon/memory_side.h>

#include <cassert>
#include <sstream>
#include <stdexcept>
#include <string>

int
main() {
  using carbon::MemorySide;

  static_assert(carbon::memory_side_count == 3);
  static_assert(static_cast<int>(MemorySide::Host) == 0);
  static_assert(static_cast<int>(MemorySide::Device) == 1);
  static_assert(static_cast<int>(MemorySide::HostDevice) == 2);

  assert(carbon::to_string(MemorySide::Host) == "host");
  assert(carbon::to_string(MemorySide::Device) == "device");
  assert(carbon::to_string(MemorySide::HostDevice) == "host_device");

  assert(carbon::to_int(MemorySide::Host) == 0);
  assert(carbon::to_int(MemorySide::Device) == 1);
  assert(carbon::to_int(MemorySide::HostDevice) == 2);

  for (const MemorySide v : carbon::memory_side_options()) {
    assert(carbon::memory_side_from_int(carbon::to_int(v)) == v);
    assert(carbon::memory_side_from_string(carbon::to_string(v)) == v);
  }

  assert(carbon::memory_side_from_string("host") == MemorySide::Host);
  assert(carbon::memory_side_from_string("HOST") == MemorySide::Host);
  assert(carbon::memory_side_from_string("Device") == MemorySide::Device);
  assert(carbon::memory_side_from_string("host_device") == MemorySide::HostDevice);
  assert(carbon::memory_side_from_string("HostDevice") == MemorySide::HostDevice);
  assert(carbon::memory_side_from_string("hostdevice") == MemorySide::HostDevice);
  assert(carbon::memory_side_from_string("host-device") == MemorySide::HostDevice);

  const auto options = carbon::memory_side_options();
  assert(options.size() == 3);
  assert(options[0] == MemorySide::Host);
  assert(options[1] == MemorySide::Device);
  assert(options[2] == MemorySide::HostDevice);

  {
    std::ostringstream os;
    os << MemorySide::Host << ',' << MemorySide::Device << ',' << MemorySide::HostDevice;
    assert(os.str() == "host,device,host_device");
  }

  assert(carbon::has_host(MemorySide::Host));
  assert(!carbon::has_host(MemorySide::Device));
  assert(carbon::has_host(MemorySide::HostDevice));

  assert(!carbon::has_device(MemorySide::Host));
  assert(carbon::has_device(MemorySide::Device));
  assert(carbon::has_device(MemorySide::HostDevice));

  bool threw = false;
  try {
    (void)carbon::memory_side_from_int(-1);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    (void)carbon::memory_side_from_int(static_cast<int>(carbon::memory_side_count));
  } catch (const std::out_of_range&) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    (void)carbon::memory_side_from_string("weird");
  } catch (const std::out_of_range&) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    (void)carbon::memory_side_from_string("");
  } catch (const std::out_of_range&) {
    threw = true;
  }
  assert(threw);

  return 0;
}
