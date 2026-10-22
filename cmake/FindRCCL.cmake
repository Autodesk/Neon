# FindRCCL
# --------
#
# Locate AMD ROCm RCCL.
#
# Sets the following variables:
#   RCCL_FOUND          - true if RCCL was found
#   RCCL_INCLUDE_DIRS   - path to rccl.h
#   RCCL_LIBRARIES      - librccl
#   RCCL_LIBRARY_DIR    - directory containing librccl
#   RCCL_VERSION        - parsed from <rccl.h> if available
#
# Also defines an imported target RCCL::rccl. RCCL is ABI-compatible with NCCL: same
# function names (ncclSend, ncclRecv, ncclComm_t, ncclResult_t, ...) but takes
# hipStream_t instead of cudaStream_t and ships its public header at
# <rccl/rccl.h>. Carbon picks RCCL automatically when CARBON_BACKEND=Sycl and
# CARBON_SYCL_TARGETS contains a hip: entry.
#
# Honors the hint variables RCCL_ROOT, RCCL_INCLUDE_DIR, RCCL_LIBRARY, and the
# environment variables RCCL_ROOT, RCCL_HOME, ROCM_PATH, ROCM_HOME.
#
# Carbon's RCCL transport requires NCCL_MAJOR>=2 (RCCL >= 2.x ships the SendRecv API).

set(_RCCL_HINT_DIRS
  ${RCCL_ROOT}
  $ENV{RCCL_ROOT}
  $ENV{RCCL_HOME}
  ${ROCM_PATH}
  $ENV{ROCM_PATH}
  $ENV{ROCM_HOME}
  /opt/rocm
)

find_path(RCCL_INCLUDE_DIR
  NAMES rccl/rccl.h rccl.h
  HINTS ${_RCCL_HINT_DIRS}
  PATH_SUFFIXES include
  DOC "Directory containing rccl/rccl.h")

find_library(RCCL_LIBRARY
  NAMES rccl
  HINTS ${_RCCL_HINT_DIRS}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
  DOC "Path to librccl")

# rccl.h #defines NCCL_MAJOR/MINOR/PATCH (matching NCCL's macro names) so the same
# parse used by FindNCCL.cmake works here. We expose them as RCCL_VERSION.
if(RCCL_INCLUDE_DIR)
  set(_rccl_header "${RCCL_INCLUDE_DIR}/rccl/rccl.h")
  if(NOT EXISTS "${_rccl_header}")
    set(_rccl_header "${RCCL_INCLUDE_DIR}/rccl.h")
  endif()
  if(EXISTS "${_rccl_header}")
    file(READ "${_rccl_header}" _rccl_header_text)
    string(REGEX MATCH "#define[ \t]+NCCL_MAJOR[ \t]+([0-9]+)" _ "${_rccl_header_text}")
    set(_RCCL_VERSION_MAJOR "${CMAKE_MATCH_1}")
    string(REGEX MATCH "#define[ \t]+NCCL_MINOR[ \t]+([0-9]+)" _ "${_rccl_header_text}")
    set(_RCCL_VERSION_MINOR "${CMAKE_MATCH_1}")
    string(REGEX MATCH "#define[ \t]+NCCL_PATCH[ \t]+([0-9]+)" _ "${_rccl_header_text}")
    set(_RCCL_VERSION_PATCH "${CMAKE_MATCH_1}")
    if(_RCCL_VERSION_MAJOR AND _RCCL_VERSION_MINOR)
      set(RCCL_VERSION "${_RCCL_VERSION_MAJOR}.${_RCCL_VERSION_MINOR}.${_RCCL_VERSION_PATCH}")
    endif()
    unset(_rccl_header_text)
  endif()
  unset(_rccl_header)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RCCL
  REQUIRED_VARS RCCL_INCLUDE_DIR RCCL_LIBRARY
  VERSION_VAR RCCL_VERSION)

if(RCCL_FOUND)
  set(RCCL_INCLUDE_DIRS "${RCCL_INCLUDE_DIR}")
  set(RCCL_LIBRARIES "${RCCL_LIBRARY}")
  get_filename_component(RCCL_LIBRARY_DIR "${RCCL_LIBRARY}" DIRECTORY)

  if(NOT TARGET RCCL::rccl)
    add_library(RCCL::rccl UNKNOWN IMPORTED)
    set_target_properties(RCCL::rccl PROPERTIES
      IMPORTED_LOCATION "${RCCL_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${RCCL_INCLUDE_DIR}")
  endif()
endif()

mark_as_advanced(RCCL_INCLUDE_DIR RCCL_LIBRARY)
