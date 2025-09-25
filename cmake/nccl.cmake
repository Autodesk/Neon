# nccl.cmake — locate a prebuilt NCCL and define target nccl::nccl
# No building is performed.
#
# Configure (set these BEFORE include(.../nccl.cmake)):
#   NCCL_REQUIRED [ON|OFF]     : fail configure if missing (default ON)
#   NCCL_ROOT                  : hint prefix (contains include/ and lib*/ under it)
#   NCCL_HOME                  : alternate hint prefix
#   NCCL_LIBRARY_DIR           : explicit directory to search for libnccl.{so,a}
#   NCCL_LIBRARY               : explicit full path to libnccl.{so,a}
#
# Outputs on success:
#   NCCL_FOUND                 : TRUE
#   NCCL_INCLUDE_DIR           : directory with nccl.h
#   NCCL_LIBRARY               : full path to libnccl
#   NCCL_VERSION_STRING        : e.g. 2.27.7
#   Imported target            : nccl::nccl

if(DEFINED _NCCL_CMAKE_INCLUDED)
    return()
endif()
set(_NCCL_CMAKE_INCLUDED TRUE)

# -----------------------------
# Defaults
# -----------------------------
if(NOT DEFINED NCCL_REQUIRED)
    set(NCCL_REQUIRED ON)
endif()

set(_NCCL_HINTS "")
list(APPEND _NCCL_HINTS
        ${NCCL_ROOT} ${NCCL_HOME}
        $ENV{NCCL_ROOT} $ENV{NCCL_HOME}
)

# Honor CMAKE_PREFIX_PATH as hints
if(CMAKE_PREFIX_PATH)
    list(APPEND _NCCL_HINTS ${CMAKE_PREFIX_PATH})
endif()

# Common prefixes (last resort)
list(APPEND _NCCL_HINTS /usr /usr/local /opt /opt/nccl)

# -----------------------------
# Try pkg-config for header/lib
# -----------------------------
set(_NCCL_FROM_PKGCFG OFF)
set(NCCL_INCLUDE_DIR "")
# Respect explicit user override for library path early
if(DEFINED NCCL_LIBRARY AND EXISTS "${NCCL_LIBRARY}")
    # keep it, will validate later
else()
    unset(NCCL_LIBRARY CACHE)
    unset(NCCL_LIBRARY)
endif()

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND AND NOT NCCL_LIBRARY)
    pkg_check_modules(_NCCL_PKG QUIET nccl)
    if(_NCCL_PKG_FOUND)
        if(_NCCL_PKG_INCLUDE_DIRS)
            list(GET _NCCL_PKG_INCLUDE_DIRS 0 NCCL_INCLUDE_DIR)
        endif()
        if(_NCCL_PKG_LIBRARY_DIRS AND NOT NCCL_LIBRARY)
            find_library(NCCL_LIBRARY
                    NAMES nccl libnccl
                    HINTS ${_NCCL_PKG_LIBRARY_DIRS}
            )
        endif()
        if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
            set(_NCCL_FROM_PKGCFG ON)
        endif()
    endif()
endif()

# -----------------------------
# Manual header search if needed
# -----------------------------
if(NOT NCCL_INCLUDE_DIR)
    find_path(NCCL_INCLUDE_DIR
            NAMES nccl.h
            HINTS ${_NCCL_HINTS}
            PATH_SUFFIXES include)
endif()

# -----------------------------
# Library search (robust)
# Order:
#   0) explicit NCCL_LIBRARY (already set)
#   1) user provided NCCL_LIBRARY_DIR (no default paths)
#   2) default system paths (catches /usr/lib on Manjaro/Arch)
#   3) expanded hints incl. CUDA toolkit dirs and common spots
# -----------------------------
# 0) explicit full path already handled above

# 1) explicit directory override
if(NOT NCCL_LIBRARY AND DEFINED NCCL_LIBRARY_DIR)
    find_library(NCCL_LIBRARY
            NAMES nccl libnccl
            PATHS "${NCCL_LIBRARY_DIR}"
            NO_DEFAULT_PATH)
endif()

# 2) default system search (no hints) — should resolve /usr/lib/libnccl.so
if(NOT NCCL_LIBRARY)
    find_library(NCCL_LIBRARY
            NAMES nccl libnccl)
endif()

# Build expanded hint list (CUDA/toolkit + common)
set(_NCCL_LIB_HINTS ${_NCCL_HINTS})

# If CUDAToolkit is present, add its likely lib locations
if(TARGET CUDA::cudart OR CUDAToolkit_FOUND)
    if(DEFINED CUDAToolkit_LIBRARY_DIR)
        list(APPEND _NCCL_LIB_HINTS
                "${CUDAToolkit_LIBRARY_DIR}"
                "${CUDAToolkit_LIBRARY_DIR}/stubs")
    endif()
    if(DEFINED CUDAToolkit_ROOT)
        list(APPEND _NCCL_LIB_HINTS
                "${CUDAToolkit_ROOT}/lib64"
                "${CUDAToolkit_ROOT}/lib"
                "${CUDAToolkit_ROOT}/targets/x86_64-linux/lib"
                "${CUDAToolkit_ROOT}/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/lib")
    endif()
    # Very common NVIDIA prefixes
    list(APPEND _NCCL_LIB_HINTS
            "/opt/cuda/lib64" "/opt/cuda/lib" "/opt/cuda/targets/x86_64-linux/lib")
    # Generic distro lib dirs (harmless duplicates OK)
    list(APPEND _NCCL_LIB_HINTS
            "/usr/lib" "/usr/lib64" "/lib" "/lib64"
            "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
            "/usr/lib/x86_64-linux-gnu"                # <--- added for Ubuntu
            "/usr/local/lib/x86_64-linux-gnu"          # <--- added for Ubuntu
    )
endif()

# Generic distro lib dirs (harmless duplicates OK)
list(APPEND _NCCL_LIB_HINTS
        "/usr/lib" "/usr/lib64" "/lib" "/lib64"
        "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}")

# 3) try again using hints explicitly
if(NOT NCCL_LIBRARY)
    find_library(NCCL_LIBRARY
            NAMES nccl libnccl
            HINTS ${_NCCL_LIB_HINTS}
            PATH_SUFFIXES lib lib64)
endif()

# -----------------------------
# Determine version from header (best-effort)
# -----------------------------
set(NCCL_VERSION_STRING "")
if(NCCL_INCLUDE_DIR)
    set(_NCCL_HEADER "${NCCL_INCLUDE_DIR}/nccl.h")
    if(EXISTS "${_NCCL_HEADER}")
        file(READ "${_NCCL_HEADER}" _nccl_h_text)
        string(REGEX MATCH "#define[ \t]+NCCL_MAJOR[ \t]+([0-9]+)" _m "${_nccl_h_text}")
        if(CMAKE_MATCH_1)
            set(_vmaj "${CMAKE_MATCH_1}")
        endif()
        string(REGEX MATCH "#define[ \t]+NCCL_MINOR[ \t]+([0-9]+)" _n "${_nccl_h_text}")
        if(CMAKE_MATCH_1)
            set(_vmin "${CMAKE_MATCH_1}")
        endif()
        string(REGEX MATCH "#define[ \t]+NCCL_PATCH[ \t]+([0-9]+)" _p "${_nccl_h_text}")
        if(CMAKE_MATCH_1)
            set(_vpat "${CMAKE_MATCH_1}")
        endif()
        if(DEFINED _vmaj AND DEFINED _vmin AND DEFINED _vpat)
            set(NCCL_VERSION_STRING "${_vmaj}.${_vmin}.${_vpat}")
        endif()
    endif()
endif()

# -----------------------------
# Finalize & report
# -----------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL
        REQUIRED_VARS NCCL_INCLUDE_DIR NCCL_LIBRARY
        VERSION_VAR NCCL_VERSION_STRING
)

if(NOT NCCL_FOUND)
    if(NCCL_REQUIRED)
        message(FATAL_ERROR
                "NCCL not found.\n"
                "Hints tried:\n"
                "  - NCCL_ROOT/NCCL_HOME/CMAKE_PREFIX_PATH\n"
                "  - Default system library paths\n"
                "  - CUDA toolkit library paths\n"
                "  - pkg-config (nccl.pc)\n"
                "Provide NCCL_ROOT, NCCL_LIBRARY_DIR, or NCCL_LIBRARY; or adjust PKG_CONFIG_PATH.")
    else()
        message(STATUS "NCCL not found (NCCL_REQUIRED=OFF). Skipping target creation.")
        return()
    endif()
endif()

# -----------------------------
# Imported target nccl::nccl
# -----------------------------
if(NOT TARGET nccl::nccl)
    # UNKNOWN avoids mislabeling static vs shared; CMake will link correctly either way.
    add_library(nccl::nccl UNKNOWN IMPORTED)
    set_target_properties(nccl::nccl PROPERTIES
            IMPORTED_LOCATION             "${NCCL_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}")
    # If CUDAToolkit target exists, expose CUDA::cudart transitively (harmless if unused)
    if(TARGET CUDA::cudart)
        set_property(TARGET nccl::nccl APPEND PROPERTY
                INTERFACE_LINK_LIBRARIES CUDA::cudart)
    endif()
endif()

# Friendly status
if(NCCL_VERSION_STRING)
    message(STATUS "Found NCCL ${NCCL_VERSION_STRING} | include: ${NCCL_INCLUDE_DIR} | lib: ${NCCL_LIBRARY}")
else()
    message(STATUS "Found NCCL | include: ${NCCL_INCLUDE_DIR} | lib: ${NCCL_LIBRARY}")
endif()
