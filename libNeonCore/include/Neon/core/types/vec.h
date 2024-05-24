#pragma once


#include "Neon/core/types/Macros.h"

#if defined(NEON_COMPILER_VS)
#pragma warning(push)
#pragma warning(disable : 4201)
#endif

#if !defined(NEON_WARP_COMPILATION)
#include "Neon/core/types/vec/vec2d_generic.h"
#endif

#include "Neon/core/types/vec/vec3d_generic.h"

#include "Neon/core/types/vec/vec4d_generic.h"


#include "Neon/core/types/vec/vecAlias.h"

#if !defined(NEON_WARP_COMPILATION)
#include "Neon/core/types/vec/vec2d_integer.tdecl.h"
#include "Neon/core/types/vec/vec2d_real.tdecl.h"
#endif

#include "Neon/core/types/vec/vec3d_integer.tdecl.h"
#include "Neon/core/types/vec/vec3d_real.tdecl.h"


#include "Neon/core/types/vec/vec4d_integer.tdecl.h"
#if !defined(NEON_WARP_COMPILATION)
#include "Neon/core/types/vec/vec4d_real.tdecl.h"
#endif


#if !defined(NEON_WARP_COMPILATION)
#include "Neon/core/types/vec/vec2d_integer.timp.h"
#include "Neon/core/types/vec/vec2d_real.timp.h"
#endif
#include "Neon/core/types/vec/vec3d_integer.timp.h"
#include "Neon/core/types/vec/vec3d_real.timp.h"
#if !defined(NEON_WARP_COMPILATION)
#include "Neon/core/types/vec/vec4d_integer.timp.h"
#include "Neon/core/types/vec/vec4d_real.timp.h"
#endif

#if defined(NEON_COMPILER_VS)
#pragma warning(pop)
#endif