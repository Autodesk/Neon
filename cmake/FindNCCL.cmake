# FindNCCL
# --------
#
# Locate NVIDIA NCCL.
#
# Sets the following variables:
#   NCCL_FOUND          - true if NCCL was found
#   NCCL_INCLUDE_DIRS   - path to nccl.h
#   NCCL_LIBRARIES      - libnccl
#   NCCL_LIBRARY_DIR    - directory containing libnccl
#   NCCL_VERSION        - parsed from <nccl.h> if available
#
# Also defines an imported target NCCL::nccl.
#
# Honors the hint variables NCCL_ROOT, NCCL_INCLUDE_DIR, NCCL_LIBRARY, and the
# environment variables NCCL_ROOT, NCCL_HOME, CUDA_PATH, CUDA_HOME.
#
# Carbon's NCCL transport requires NCCL >= 2.7 (introduces ncclSend/ncclRecv).

set(_NCCL_HINT_DIRS
  ${NCCL_ROOT}
  $ENV{NCCL_ROOT}
  $ENV{NCCL_HOME}
  ${CUDAToolkit_ROOT}
  $ENV{CUDA_PATH}
  $ENV{CUDA_HOME}
  /usr/local/cuda
)

find_path(NCCL_INCLUDE_DIR
  NAMES nccl.h
  HINTS ${_NCCL_HINT_DIRS}
  PATH_SUFFIXES include
  DOC "Directory containing nccl.h")

find_library(NCCL_LIBRARY
  NAMES nccl
  HINTS ${_NCCL_HINT_DIRS}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
  DOC "Path to libnccl")

if(NCCL_INCLUDE_DIR AND EXISTS "${NCCL_INCLUDE_DIR}/nccl.h")
  file(READ "${NCCL_INCLUDE_DIR}/nccl.h" _nccl_header)
  string(REGEX MATCH "#define[ \t]+NCCL_MAJOR[ \t]+([0-9]+)" _ "${_nccl_header}")
  set(_NCCL_VERSION_MAJOR "${CMAKE_MATCH_1}")
  string(REGEX MATCH "#define[ \t]+NCCL_MINOR[ \t]+([0-9]+)" _ "${_nccl_header}")
  set(_NCCL_VERSION_MINOR "${CMAKE_MATCH_1}")
  string(REGEX MATCH "#define[ \t]+NCCL_PATCH[ \t]+([0-9]+)" _ "${_nccl_header}")
  set(_NCCL_VERSION_PATCH "${CMAKE_MATCH_1}")
  if(_NCCL_VERSION_MAJOR AND _NCCL_VERSION_MINOR)
    set(NCCL_VERSION "${_NCCL_VERSION_MAJOR}.${_NCCL_VERSION_MINOR}.${_NCCL_VERSION_PATCH}")
  endif()
  unset(_nccl_header)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL
  REQUIRED_VARS NCCL_INCLUDE_DIR NCCL_LIBRARY
  VERSION_VAR NCCL_VERSION)

if(NCCL_FOUND)
  set(NCCL_INCLUDE_DIRS "${NCCL_INCLUDE_DIR}")
  set(NCCL_LIBRARIES "${NCCL_LIBRARY}")
  get_filename_component(NCCL_LIBRARY_DIR "${NCCL_LIBRARY}" DIRECTORY)

  if(NOT TARGET NCCL::nccl)
    add_library(NCCL::nccl UNKNOWN IMPORTED)
    set_target_properties(NCCL::nccl PROPERTIES
      IMPORTED_LOCATION "${NCCL_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}")
  endif()

  if(NCCL_VERSION AND NCCL_VERSION VERSION_LESS "2.7")
    message(FATAL_ERROR
      "Carbon requires NCCL >= 2.7 (ncclSend/ncclRecv). Found NCCL ${NCCL_VERSION} at ${NCCL_LIBRARY}.")
  endif()
endif()

mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
