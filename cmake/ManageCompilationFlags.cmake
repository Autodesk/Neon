# a. Reset some CMAKE defaults (like CMAKE_CXX_FLAGS_DEBUG_INIT)
# b. Define NeonCXXFlags and NeonCUDAFlags

# Enabling interprocedural optimizations (LTO/IPO)
include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if (result)
    set_property(TARGET NeonDeveloperLib PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else ()
    message(WARNING "IPO is not supported: ${output}")
endif ()

# Enabling debug symbols on Debug
set(CMAKE_CXX_FLAGS_DEBUG_INIT
        $<$<CXX_COMPILER_ID:GNU>:-O0 -g3>
        $<$<CXX_COMPILER_ID:Clang>:-O0 -g3>)

set(CMAKE_CUDA_FLAGS_INIT
        $<$<CXX_COMPILER_ID:GNU>:-O0 -G>
        $<$<CXX_COMPILER_ID:Clang>:-O0 -G>)


#Interface library that contains compilation flags and other common stuff needed by all Neon's targets
# CUDA and C++ compiler flags
set(NeonCXXFlags
        $<$<CXX_COMPILER_ID:GNU>:$<$<CONFIG:Debug>:-O0 -g3>>
        $<$<CXX_COMPILER_ID:Clang>:$<$<CONFIG:Debug>:-O0 -g3>>
        $<$<CXX_COMPILER_ID:GNU>:$<$<CONFIG:Release>:-O3 -Wno-deprecated-declarations>>
        $<$<CXX_COMPILER_ID:Clang>:$<$<CONFIG:Release>:-O3 -Wno-deprecated-declarations>>
        #Add MSVC-specific compiler flags here
        # Ignore "Conditional expression is constant" warning until NVCC support c++17. Once that happens we can change the code to use if constexpr
        # Ignore "OpenMP Collapse warning until VS supports full OpenMP functionality
        # Ignore "unreferenced local function has been removed" that shows up when we include cublas_v2 header
        $<$<CXX_COMPILER_ID:MSVC>:-D_SCL_SECURE_NO_WARNINGS /openmp /std:c++17 /MP /W4 /WX /wd4849 /wd4127 /wd4996 /wd4505 /wd4702 /bigobj>

        #Add GCC specific compiler flags here
		#-Wno-class-memaccess for "writing to an object of type XXX with no trivial copy-assignment; use copy-assignment or copy-initialization instead"
        $<$<CXX_COMPILER_ID:GNU>:-m64 -Wall -Wextra -Werror -Wno-unused-function -Wno-deprecated-declarations -Wno-class-memaccess -Wno-deprecated-declarations>

        #Add Clang specific compiler flags here
        $<$<CXX_COMPILER_ID:Clang>:-m64 -Wall -Wextra -Werror -Wno-unused-function -Wno-deprecated-declarations -Wno-deprecated-copy -Wno-unused-parameter -Wno-unused-private-field -Wno-braced-scalar-init -Wno-unused-variable -Wno-unused-but-set-variable -Wno-deprecated-declarations >
        )

set(MSVC_XCOMPILER_FLAGS "/openmp /std:c++17  /bigobj")
set(NeonCUDAFlags
        # Optimization flags for Release
        $<$<CXX_COMPILER_ID:GNU>: $<$<CONFIG:Release>:-O3> >
        $<$<CXX_COMPILER_ID:Clang>: $<$<CONFIG:Release>:-O3> >
        # Optimization flags for Debug
        $<$<CXX_COMPILER_ID:GNU>: $<$<CONFIG:Debug>:-O0 -G> >
        $<$<CXX_COMPILER_ID:Clang>: $<$<CONFIG:Debug>:-O0 -G> >
        # Host compiler
        $<$<CXX_COMPILER_ID:GNU>:-Xcompiler -fopenmp -std=c++17 $<$<CONFIG:Release>:-O3> $<$<CONFIG:Debug>:-O0> >
        $<$<CXX_COMPILER_ID:Clang>:-Xcompiler -fopenmp -std=c++17 $<$<CONFIG:Release>:-O3> $<$<CONFIG:Debug>:-O0>>
        $<$<CXX_COMPILER_ID:MSVC>:-Xcompiler ${MSVC_XCOMPILER_FLAGS}>
        #Disables warning
        #177-D "function XXX was declared but never referenced"
        -Xcudafe "--display_error_number --diag_suppress=177"
        -lineinfo
        --expt-extended-lambda
        -use_fast_math
        --expt-relaxed-constexpr
        -Xptxas -warn-spills -res-usage
        --ptxas-options=-v
        --relocatable-device-code=true
        )
