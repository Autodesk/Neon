#pragma once
#include "Neon/core/core.h"

#ifdef NEON_COMPILER_CUDA
#if (__CUDA_ARCH__ >= 350)
#define NEON_CUDA_CONST_LOAD(ADDR) (__ldg(((ADDR))))
#else
#define NEON_CUDA_CONST_LOAD(ADDR) (*((ADDR)))
#endif
#else
#define NEON_CUDA_CONST_LOAD(ADDR) (*((ADDR)))
#endif
