#include <carbon/carbon.h>

int
main() {
  carbon::init();
  CARBON_LOG_INFO("carbon logging smoke test");
  return 0;
}
