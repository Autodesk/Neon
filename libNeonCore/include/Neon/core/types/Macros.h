#pragma once
/*
 * Important Compiler preprocessor variables:
 * __CUDACC__ defined by acc nvidia compiler
 * _WIN32     defined by VisualStudio
 * __GNUG__   defined by gcc and icc
 * __clang__  defined by clang
 * __INTEL_COMPILER
 */

#if defined(__CUDA_ARCH__)
#define NEON_PLACE_CUDA_DEVICE
#else
#define NEON_PLACE_CUDA_HOST
#endif

#ifdef __CUDACC__
#define NEON_COMPILER_CUDA
#define NEON_COMPILER "CUDA"
#endif

#if defined(__NVCC__) && !defined(NEON_COMPILER)
#define NEON_COMPILER_NVCC
#define NEON_COMPILER "NVCC"
#endif

#ifdef __INTEL_COMPILER
#define NEON_COMPILER_ICC
#define NEON_COMPILER "ICC"
#endif

#if defined(_WIN32) && !defined(NEON_COMPILER)
#define NEON_COMPILER_VS
#define NEON_COMPILER "VS"
#endif

#if defined(__GNUG__) && !defined(NEON_COMPILER)
#define NEON_COMPILER_GCC
#define NEON_COMPILER "GCC"
#endif

#if defined(__clang__) && !defined(NEON_COMPILER)
#define NEON_COMPILER_CLANG
#define NEON_COMPILER "Clang"
#endif

#if !defined(NEON_COMPILER)
#error Compiler framework not supported. Supported options are:  nvcc, icc, vs, gcc, clang.
#endif

/*****************
* This File contains a set of preprocessor macros to better handle mix HOST and DEVICE
* code in the CUDA framework. Similar technique has also been used by NVIDIA in the HEMI project.
* Hemi project is store at https://github.com/harrism/hemi
*
* While we could have used it, the project itself seams too complex for our needs.
* We will use the same technique to activate/deactivate flags like __device__ and __host__
*
* Some useful roles
*
* 1. If the function is going to be called by both device and host, then one of the following flags:
*			a. DRMR_DEV_CALLABLE
*			b. DRMR_DEV_CALLABLE_MEMBER
*			c. DRMR_DEV_CALLABLE_INLINE
*			d. DRMR_DEV_CALLABLE_INLINE_MEMBER
*
* 2. If the function is CUDA KERNEL function use the following flag:
*			a. DRMR_LAUNCHABLE
*
* 3. If in a cpp file shared between host and device you want to have a section recognized only the CUDA compiler,
*    then use the following macro that is defined only when NVCC is the active compiler.
*			a. DRMR_DEVICE_ONLY_SECTION
*
* 4. We don't export any macro for defining code sections that have to be activated only with an host compiler as
*    NVCC is already dropping all the code that does not have any "DEVICE" label.
*
*****************/
#pragma once

#if defined(NEON_COMPILER_CUDA)
//------------------------------------------------------------------------------------
///////////////// NVCC COMPILER IS ACTIVE AT THE MOMENT //////////////////////////////
#define NEON_CUDA_KERNEL __global__
#define NEON_CUDA_DEVICE_ONLY __device__
#define NEON_CUDA_HOST_ONLY __host__
#define NEON_CUDA_HOST_DEVICE __host__ __device__
#define NEON_CUDA_DEVICE_ONLY_SECTION
#define NEON_CUDA_HOST_DEVICE_CLASS
#define NEON_DEVICE_ONLY_SECTION 1
#else  //// defined(__CUDACC__)

///////////////// A NON NVCC COMPILER IS ACTIVE AT THE MOMENT ////////////////////////
#define NEON_CUDA_KERNEL
#define NEON_CUDA_DEVICE_ONLY
#define NEON_CUDA_HOST_ONLY
#define NEON_CUDA_HOST_DEVICE
#define NEON_CUDA_DEVICE_ONLY_SECTION
#define NEON_CUDA_HOST_DEVICE_CLASS
//// Let's be sure that DRMR_DEV_ONLY_SECTION is not defined when
//// NVCC is not the active compiler
#if defined(NEON_DEVICE_ONLY_SECTION)
#undef NEON_DEVICE_ONLY_SECTION
#endif
#define NEON_DEVICE_ONLY_SECTION 0

#endif  //// defined(__CUDACC__)
//------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------
///////////////// MACRO TO IDENTIFY IN/OUT/IN-OUT/SUPPORT parameters /////////////////
//// _IN_PARAM_ -> the parameter is read only.
#define NEON_IN
//// _OUT_PARAM_ -> the parameter is used to store the result.
#define NEON_OUT
#define NEON_IN_HALO
//// _INOUT_PARAM_ -> the parameter is used both as input and output.
#define NEON_IO
//// _SUPPORT_PARAM_ -> the parameter is not input or output. It is used to store partial results.
#define NEON_SUP
#define NEON_TMP
/// NEON_ITMP -> buffer used for input first but also as tmp buffer
#define NEON_ITMP
//------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------
#if 0
//////// SUPPORT_FOR VARIADIC MACORS BOTH FOR BOTH GCC, CLANG and VS /////////////////
//#define ___NEON_MEM_GLUE(x, y) x y
//
//#define ___NEON_MEM_RETURN_ARG_COUNT(_1_, _2_, _3_, _4_, _5_, count, ...) count
//#define ___NEON_MEM_EXPAND_ARGS(args) ___NEON_MEM_RETURN_ARG_COUNT args
//#define ___NEON_MEM_COUNT_ARGS_MAX5(...) ___NEON_MEM_EXPAND_ARGS((__VA_ARGS__, 5, 4, 3, 2, 1, 0))
//
//#define ___NEON_MEM_OVERLOAD_MACRO22(name, count) name##count
//#define ___NEON_MEM_OVERLOAD_MACRO21(name, count) ___NEON_MEM_OVERLOAD_MACRO22(name, count)
//#define ___NEON_MEM_OVERLOAD_MACRO2(name, count) ___NEON_MEM_OVERLOAD_MACRO21(name, count)
//
//#define ___NEON_MEM_CALL_OVERLOAD(name, ...)
//        ___NEON_MEM_GLUE(___NEON_MEM_OVERLOAD_MACRO2(name, ___NEON_MEM_COUNT_ARGS_MAX5(__VA_ARGS__)), (__VA_ARGS__))
#endif


//https://stackoverflow.com/questions/11761703/overloading-macro-on-number-of-arguments
// #define NEON_EXPAND(x) x
//#define _NEON_OVERLOADED_MACRO(M, ...) __NEON_OVR(M, __NEON_COUNT_ARGS(__VA_ARGS__)) (__VA_ARGS__)
//#define __NEON_OVR(macroName, number_of_args)   __NEON_OVR_EXPAND(macroName, number_of_args)
//#define __NEON_OVR_EXPAND(macroName, number_of_args)    macroName##number_of_args
//
//#define __NEON_COUNT_ARGS(...)  __NEON_ARG_PATTERN_MATCH(__VA_ARGS__, 9,8,7,6,5,4,3,2,1)
//#define __NEON_ARG_PATTERN_MATCH(_1,_2,_3,_4,_5,_6,_7,_8,_9, N, ...)   N

#define __NEON_BUGFX(x) x

#define __NEON_NARG2(...) __NEON_BUGFX(__NEON_NARG1(__VA_ARGS__, __NEON_RSEQN()))
#define __NEON_NARG1(...) __NEON_BUGFX(__NEON_ARGSN(__VA_ARGS__))
#define __NEON_ARGSN(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#define __NEON_RSEQN() 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

#define __NEON_FUNC2(name, n) name##n
#define __NEON_FUNC1(name, n) __NEON_FUNC2(name, n)
#define _NEON_OVERLOADED_MACRO(func, ...)                       \
    __NEON_FUNC1(func, __NEON_BUGFX(__NEON_NARG2(__VA_ARGS__))) \
    (__VA_ARGS__)

//------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------
///////////////// __attribute__((unused)) /////////////////
#ifdef NEON_COMPILER_VS
#define NEON_ATTRIBUTE_UNUSED
#endif

#ifdef NEON_COMPILER_CLANG
#define NEON_ATTRIBUTE_UNUSED __attribute__((unused))
#endif

#ifdef NEON_COMPILER_GCC
#define NEON_ATTRIBUTE_UNUSED __attribute__((unused))
#endif

#ifdef NEON_COMPILER_ICC
#define NEON_ATTRIBUTE_UNUSED __attribute__((unused))
#endif

#ifdef NEON_COMPILER_CUDA
#define NEON_ATTRIBUTE_UNUSED
#endif

//------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------
///////////////// RESTRICT /////////////////
#ifdef NEON_COMPILER_VS
#define NEON_RESTRICT __restrict
#endif
/// gcc -E -dM - < /dev/null
#ifdef NEON_COMPILER_GCC
#define NEON_RESTRICT __restrict__
#endif
/// icc -E -dM - < /dev/null
/// -restrict must be used at compilation
#ifdef NEON_COMPILER_ICC
#define NEON_RESTRICT restrict
#endif

#if defined(NEON_COMPILER_CUDA)
#if!defined(_WIN32)
#define NEON_RESTRICT __restrict__
#else
#define NEON_RESTRICT
#endif
#endif

#ifdef NEON_COMPILER_CLANG
#define NEON_RESTRICT __restrict__
#endif
//------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------
///////////////// __attribute__((__deprecated__)) /////////////////
#ifdef NEON_COMPILER_VS
#define NEON_DEPRECATED __declspec(deprecated)
#endif

#ifdef NEON_COMPILER_CLANG
#define NEON_DEPRECATED __attribute__((__deprecated__))
#endif

#ifdef NEON_COMPILER_GCC
#define NEON_DEPRECATED __attribute__((__deprecated__))
#endif

#ifdef NEON_COMPILER_ICC
#define NEON_DEPRECATED __attribute__((__deprecated__))
#endif

#ifdef NEON_COMPILER_CUDA
#define NEON_DEPRECATED
#endif

//------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------
#if defined(__linux__) || defined(__APPLE__)
#define NEON_FUNCTION_NAME() __PRETTY_FUNCTION__
#elif defined(_WIN64)
#define NEON_FUNCTION_NAME() __FUNCTION__
#else
#error Undetected OS! Check _WIN64, __linux__ or __APPLE__ flags.
#endif

#if defined(__linux__)
#define NEON_OS_LINUX
#elif defined(_WIN64)
#define NEON_OS_WINDOWS
#elif defined(__APPLE__)
#define NEON_OS_MAC
#else
#error Undetected OS! Check _WIN64, __linux__ or __APPLE__ flags.
#endif


#define NEON_CUDA_CHECK_LAST_ERROR                            \
    {                                                         \
        cudaDeviceSynchronize();                              \
        cudaError_t error = cudaPeekAtLastError();            \
        if (error != cudaSuccess) {                           \
            Neon::NeonException exc;                          \
            exc << "\n Error: " << cudaGetErrorString(error); \
            NEON_THROW(exc);                                  \
        }                                                     \
    }

#ifndef __LONG_MAX__
#define __LONG_MAX__ 9223372036854775807L
#endif

#ifndef __INT_MAX_
#define __INT_MAX_ 2147483647
#endif

#ifndef NEON_DIVIDE_UP
#define NEON_DIVIDE_UP(num, divisor) (num + divisor - 1) / (divisor)
#endif