#pragma once

// Include all Dense grid types to ensure they are complete
#include "Neon/domain/details/Dense/Idx.h"
#include "Neon/domain/details/Dense/Span.h"
#include "Neon/domain/details/Dense/Partition.h"
#include "Neon/domain/details/Dense/Field.h"
#include "Neon/domain/details/Dense/Grid.h"
#include "Neon/domain/interface/GridConcept.h"

namespace Neon::domain::details::Dense {

/**
 * This file contains static assertions to validate that all Dense grid types
 * satisfy their respective concepts. These assertions are placed here to ensure
 * all types are complete when the concept checks are evaluated.
 */

// Validate that Dense::Idx satisfies the Neon::Idx concept
static_assert(Neon::Idx<Idx>, "Dense::Idx must satisfy the Neon::Idx concept");

// Validate that Dense::Span satisfies the Neon::Span concept for different layouts
static_assert(Neon::Span<Span<0>>, "Dense::Span<0> must satisfy the Neon::Span concept");
static_assert(Neon::Span<Span<1>>, "Dense::Span<1> must satisfy the Neon::Span concept");

// Validate that Dense::Partition satisfies the Neon::Partition concept for different instantiations
static_assert(Neon::Partition<Partition<0, int, 0>>, "Dense::Partition<0, int, 0> must satisfy the Neon::Partition concept");
static_assert(Neon::Partition<Partition<0, float, 1>>, "Dense::Partition<0, float, 1> must satisfy the Neon::Partition concept");
static_assert(Neon::Partition<Partition<1, double, 2>>, "Dense::Partition<1, double, 2> must satisfy the Neon::Partition concept");

// Validate that Dense::Field satisfies the Neon::Field concept for different instantiations
static_assert(Neon::Field<Field<0, int, 0>>, "Dense::Field<0, int, 0> must satisfy the Neon::Field concept");
static_assert(Neon::Field<Field<0, float, 1>>, "Dense::Field<0, float, 1> must satisfy the Neon::Field concept");
static_assert(Neon::Field<Field<1, double, 2>>, "Dense::Field<1, double, 2> must satisfy the Neon::Field concept");

// Validate that Dense::Grid satisfies the Neon::domain::Grid concept for different layouts
static_assert(Neon::domain::Grid<Grid<0>>, "Dense::Grid<0> must satisfy the Neon::domain::Grid concept");
static_assert(Neon::domain::Grid<Grid<1>>, "Dense::Grid<1> must satisfy the Neon::domain::Grid concept");

} // namespace Neon::domain::details::Dense
