#pragma once


#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __shared__
#define __constant__
#define __global__

// This is slightly mental, but gets it to properly index device function calls like __popc and whatever.
#define __CUDACC__
#include <device_functions.h>

// These headers are all implicitly present when you compile CUDA with clang. Clion doesn't know that, so
// we include them explicitly to make the indexer happy. Doing this when you actually build is, obviously,
// a terrible idea :D
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_cmath.h>
#include <__clang_cuda_complex_builtins.h>
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_math_forward_declares.h>
#endif  // __JETBRAINS_IDE__


#include "Neon/core/types/Access.h"
#include "Neon/core/types/BasicTypes.h"
#include "Neon/core/types/chrono.h"
#include "Neon/core/types/Exceptions.h"
#include "Neon/core/types/Macros.h"
#include "Neon/core/types/mode.h"
#include "Neon/core/types/SetIdx.h"
#include "Neon/core/types/vec.h"
#include "Neon/core/types/DataUse.h"
#include "Neon/core/types/memOptions.h"
#include "Neon/core/types/DeviceType.h"
#include "Neon/core/types/Allocator.h"
#include "Neon/core/types/DataView.h"

#include "Neon/core/tools/development/workInProgress.h"
#include "Neon/core/tools/Logger.h"
#include "Neon/core/tools/metaprogramming.h"
#include "Neon/core/tools/Report.h"
