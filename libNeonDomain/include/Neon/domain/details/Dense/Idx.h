#pragma once

#include "Neon/core/core.h"
#include "Neon/domain/interface/GridConcept.h"

namespace Neon::domain::details::Dense {

// Common forward declarations
// template <int Layout>
// class Grid;
//
// template <int Layout>
// class Span;
//
// template <int Layout, typename T, int C>
// class Partition;

/**
 * Dense Grid Index type - satisfies the Neon::Idx concept
 * Used for indexing elements within Dense grid partitions
 */
struct Idx
{
    template <int Layout, typename T, int C>
    friend class Partition;
    template <int Layout>
    friend class Span;
    template <int Layout>
    friend class Grid;

    template <int Layout, typename T, int C>
    friend class Field;

    // Dense grid specific types
    using Offset = int32_t;
    using Location = index_3d;
    using Count = int32_t;

    // Default constructor (required by Neon::Idx concept)
    Idx() = default;

    // Copy constructor (required by Neon::Idx concept)
    Idx(const Idx&) = default;

    // Assignment operator
    Idx& operator=(const Idx&) = default;

    // Location storage
    Location mLocation = 0;

    // Constructors for creating index with specific coordinates
    NEON_CUDA_HOST_DEVICE inline explicit Idx(const Location::Integer& x,
                                              const Location::Integer& y,
                                              const Location::Integer& z);

    NEON_CUDA_HOST_DEVICE inline explicit Idx(const Location& location);

    // Accessor methods
    NEON_CUDA_HOST_DEVICE inline auto setLocation() -> Location&;

    NEON_CUDA_HOST_DEVICE inline auto getLocation() const -> const Location&;
};

// Note: Static assertion for concept compliance moved to avoid incomplete type issues

}  // namespace Neon::domain::details::Dense

#include "Idx.imp.h"
