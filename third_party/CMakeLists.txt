# Vendored third-party dependencies used by carbon's examples.
#
# Currently:
#   - clipp (header-only, MIT) — same single-header CLI parser Neon uses for its
#     example/benchmark binaries; carbon vendors a verbatim copy under clipp/clipp.h.
#
# Each dependency is exposed as an INTERFACE library so consumer targets only need
# target_link_libraries(<tgt> PRIVATE carbon::third_party::<name>) to pick up the
# include path. None of these are part of carbon's public API and they are only
# pulled in when CARBON_BUILD_EXAMPLES is on.

add_library(carbon_third_party_clipp INTERFACE)
add_library(carbon::third_party::clipp ALIAS carbon_third_party_clipp)
target_include_directories(carbon_third_party_clipp INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/clipp")
