#pragma once

#if defined(_WIN32)
#  if defined(CARBON_BUILD_SHARED)
#    if defined(carbon_EXPORTS)
#      define CARBON_API __declspec(dllexport)
#    else
#      define CARBON_API __declspec(dllimport)
#    endif
#  else
#    define CARBON_API
#  endif
#else
#  define CARBON_API __attribute__((visibility("default")))
#endif
