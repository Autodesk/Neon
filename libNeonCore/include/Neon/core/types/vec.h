#pragma once


#include "Neon/core/types/Macros.h"

#if defined(NEON_COMPILER_VS)
#pragma warning(push)
#pragma warning(disable : 4201)
#endif

#include "Neon/core/types/vec/vec2d_generic.h"
#include "Neon/core/types/vec/vec3d_generic.h"
#include "Neon/core/types/vec/vec4d_generic.h"

#include "Neon/core/types/vec/vecAlias.h"

#include "Neon/core/types/vec/vec2d_integer.tdecl.h"
#include "Neon/core/types/vec/vec2d_real.tdecl.h"
#include "Neon/core/types/vec/vec3d_integer.tdecl.h"
#include "Neon/core/types/vec/vec3d_real.tdecl.h"
#include "Neon/core/types/vec/vec4d_integer.tdecl.h"
#include "Neon/core/types/vec/vec4d_real.tdecl.h"

#include "Neon/core/types/vec/vec2d_integer.timp.h"
#include "Neon/core/types/vec/vec2d_real.timp.h"
#include "Neon/core/types/vec/vec3d_integer.timp.h"
#include "Neon/core/types/vec/vec3d_real.timp.h"
#include "Neon/core/types/vec/vec4d_integer.timp.h"
#include "Neon/core/types/vec/vec4d_real.timp.h"

#if defined(NEON_COMPILER_VS)
#pragma warning(pop)
#endif