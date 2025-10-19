#include "Neon/domain/details/Dense/Partition.h"
#include "Neon/domain/details/Dense/Span.h"
#include "Neon/domain/details/Dense/Field.h"
#include "Neon/domain/details/Dense/Grid.h"
#include "Neon/domain/details/Dense/Layout.h"
#include "Neon/domain/details/Dense/ConceptValidation.h"

namespace Neon::domain::details::Dense {

// ConceptValidation.h is included to trigger static assertions
// that validate all Dense types satisfy their respective concepts
// Layout.h provides CLI utilities for layout options

// Example Grid instantiation to ensure compilation
Grid<0> exampleGrid;


} // namespace Neon::domain::details::Dense