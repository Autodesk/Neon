#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/DataUse.h"
#include "Neon/set/MemoryOptions.h"
#include "Neon/set/DataSet.h"
#include "Neon/set/Containter.h"
#include "Neon/set/StencilSemantic.h"
#include "Neon/set/TransferMode.h"
#include "Neon/core/types/Execution.h"
#include "Neon/set/ExecutionThreadSpan.h"
#include <concepts>
#include <type_traits>

namespace Neon {

/**
 * Concept for Index types used for grid element indexing
 */
template <typename I>
concept Idx = std::is_default_constructible_v<I> && std::is_copy_constructible_v<I>;

/**
 * Concept for Span types used for computation/iteration space (partition table mechanism)
 */
template <typename S>
concept Span = std::is_default_constructible_v<S> && std::is_copy_constructible_v<S> && requires {
    // Span must define execution thread span type
    { S::executionThreadSpan } -> std::same_as<const Neon::set::details::ExecutionThreadSpan&>;
    
    // Span must define execution thread span index type
    typename S::ExecutionThreadSpanIndexType;
    
    // Span must define Idx type that satisfies Idx concept
    requires Idx<typename S::Idx>;
} && requires(S span, typename S::Idx& idx, 
              const std::make_unsigned_t<typename S::ExecutionThreadSpanIndexType>& x,
              const std::make_unsigned_t<typename S::ExecutionThreadSpanIndexType>& y, 
              const std::make_unsigned_t<typename S::ExecutionThreadSpanIndexType>& z) {
    // Span must provide setAndValidate method for 3D indexing
    {
        span.setAndValidate(idx, x, y, z)
    } -> std::same_as<bool>;
};

/**
 * Concept for Partition types that define the required nested types and methods
 */
template <typename P>
concept Partition = requires {
    // Partition must define Type for stored data
    typename P::Type;
    
    // Partition must define Idx type that satisfies Idx concept
    requires Idx<typename P::Idx>;
    
    // Partition must define Span type that satisfies Span concept
    requires Span<typename P::Span>;
    
    // Partition must define neighbor data types
    typename P::NghData;
    typename P::NghIdx;
    
    // Partition must define Cardinality
    static_cast<int>(P::Cardinality);
} && requires(P partition, const P const_partition, typename P::Idx idx, typename P::NghIdx nghIdx, int cardinalityIdx) {
    // Partition ID accessor
    { const_partition.prtID() } -> std::same_as<int>;
    
    // Cardinality accessor
    { const_partition.cardinality() } -> std::same_as<int>;
    
    // Dimension accessor  
    { const_partition.dim() } -> std::same_as<const Neon::index_3d>;
    
    // Origin accessor
    { const_partition.origin() } -> std::same_as<const Neon::index_3d>;
    
    // Domain size accessor
    { const_partition.getDomainSize() } -> std::same_as<Neon::index_3d>;
    
    // Global index computation
    { const_partition.getGlobalIndex(idx) } -> std::same_as<Neon::index_3d>;
    
    // Neighbor data access
    { const_partition.getNghData(idx, nghIdx, cardinalityIdx) } -> std::same_as<typename P::NghData>;
    
    // Data element access (const)
    { const_partition(idx, cardinalityIdx) } -> std::same_as<const typename P::Type&>;
    
    // Data element access (mutable)
    { partition(idx, cardinalityIdx) } -> std::same_as<typename P::Type&>;
};

/**
 * Concept for Field types that define the required nested types and methods
 */
template <typename F>
concept Field = requires {
    // Field must define neighbor data and index types
    typename F::NghData;
    typename F::NghIdx;
    
    // Field must define Type for stored data
    typename F::Type;
    
    // Field must define PartitionTable type for data distribution management
    typename F::PartitionTable;
    
    // Field must define Idx type that satisfies Idx concept
    requires Idx<typename F::Idx>;
    
    // Field must define Span type that satisfies Span concept
    requires Span<typename F::Span>;
    
    // Field must define Partition type that satisfies Partition concept
    requires Partition<typename F::Partition>;
} && requires(F field) {
    // Fields must support halo update operations
    {
        field.newHaloUpdate(Neon::set::StencilSemantic::standard,
                           Neon::set::TransferMode::get,
                           Neon::Execution::device)
    } -> std::same_as<Neon::set::Container>;
    
    // Fields must provide access to partition table
    {
        field.getPartitionTable()
    } -> std::convertible_to<const typename F::PartitionTable&>;
};

}  // namespace Neon

namespace Neon::domain {

/**
 * Concept for sparsity pattern functions that take a 3D index and return bool
 */
template <typename SP>
concept SparsityPattern = requires(SP sparsityPattern) {
    {
        sparsityPattern(Neon::index_3d(0, 0, 0))
    } -> std::same_as<bool>;
};

/**
 * ═══════════════════════════════════════════════════════════════════════════════════════
 * 
 *                                    GRID CONCEPT
 *                           Complete Type Hierarchy & Interface
 * 
 * ═══════════════════════════════════════════════════════════════════════════════════════
 * 
 * The Grid concept defines a complete computational domain with the following type hierarchy:
 * 
 *   ┌─────────────────────────────────────────────────────────────────────────────────────┐
 *   │                                     Grid                                            │
 *   │                                   • SpanTable                                      │
 *   │ ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  ┌───────────────────────┐    │
 *   │ │     Idx     │  │    Span     │  │   Partition<T,C>│  │     Field<T,C>        │    │
 *   │ │             │  │             │  │                 │  │                       │    │
 *   │ │ • default   │  │ • Idx ──────┼─►│ • Idx ──────────┼─►│ • Idx ────────────────┼──┐ │
 *   │ │   construct │  │ • execution │  │ • Span ─────────┼─►│ • Span ───────────────┼──┤ │
 *   │ │ • copy      │  │   ThreadSpan│  │ • Type, NghData │  │ • Partition ──────────┼──┤ │
 *   │ │   construct │  │ • setAndVal │  │ • Cardinality   │  │ • Type, PartitionTable│  │ │
 *   │ │             │  │   idate()   │  │ • prtID(), dim()│  │ • NghData, NghIdx     │  │ │
 *   │ │             │  │             │  │ • operator()    │  │ • newHaloUpdate()     │  │ │
 *   │ └─────────────┘  └─────────────┘  └─────────────────┘  └───────────────────────┘  │ │
 * └─────────────────────────────────────────────────────────────────────────────────────┘ │
 *   │                                                                                     │
 *   └─────────────────────────────────────────────────────────────────────────────────────┘
 * 
 * TYPE RELATIONSHIPS & CONSISTENCY:
 * 
 * • Grid::Idx ≡ Span::Idx ≡ Partition::Idx ≡ Field::Idx
 *   └─ All index types must be identical across the hierarchy
 * 
 * • Grid::Span ≡ Partition::Span ≡ Field::Span  
 *   └─ All span types must be identical across the hierarchy
 * 
 * • Grid::Partition<T,C> ≡ Field::Partition
 *   └─ Field partition type must match Grid partition instantiation
 * 
 * CONCEPT VALIDATION HIERARCHY:
 * 
 * Grid ──┬─► Idx        → must satisfy Neon::Idx concept
 *        ├─► Span       → must satisfy Neon::Span concept  
 *        │   └─► Idx    → must satisfy Neon::Idx concept
 *        ├─► Partition  → must satisfy Neon::Partition concept
 *        │   ├─► Idx   → must satisfy Neon::Idx concept  
 *        │   └─► Span  → must satisfy Neon::Span concept
 *        └─► Field      → must satisfy Neon::Field concept
 *            ├─► Idx        → must satisfy Neon::Idx concept
 *            ├─► Span       → must satisfy Neon::Span concept
 *            └─► Partition  → must satisfy Neon::Partition concept
 * 
 * FUNCTIONAL REQUIREMENTS:
 * 
 * Grid Interface:
 * • newField<T,C>(name, cardinality, defaultValue, ...) → Field<T,C>
 * • getSpan(execution, setIdx, dataView) → const Span&
 * • getSpanTable() → const SpanTable&
 * 
 * Field Interface:
 * • newHaloUpdate(semantic, transferMode, execution) → Container
 * • getPartitionTable() → const PartitionTable&
 * 
 * ═══════════════════════════════════════════════════════════════════════════════════════
 */
template <typename G>
concept Grid = requires {
    // ═══════════════════════════════════════════════════════════════════════════════════
    // TYPE EXISTENCE & CONCEPT COMPLIANCE
    // ═══════════════════════════════════════════════════════════════════════════════════
    
    // Grid must define Idx type that satisfies Idx concept
    requires Neon::Idx<typename G::Idx>;
    
    // Grid must define Span type that satisfies Span concept
    requires Neon::Span<typename G::Span>;
    
    // Grid must define SpanTable type for span management
    typename G::SpanTable;
    
    // Grid must define Partition template alias that satisfies Partition concept
    requires Neon::Partition<typename G::template Partition<int, 0>>;
    requires Neon::Partition<typename G::template Partition<double, 1>>;
    
    // Grid must define Field template alias that satisfies Field concept
    requires Neon::Field<typename G::template Field<int, 0>>;
    requires Neon::Field<typename G::template Field<double, 1>>;
    
    // ═══════════════════════════════════════════════════════════════════════════════════
    // TYPE CONSISTENCY REQUIREMENTS
    // ═══════════════════════════════════════════════════════════════════════════════════
    
    // All Idx types must be identical across the hierarchy
    requires std::same_as<typename G::Idx, typename G::Span::Idx>;
    requires std::same_as<typename G::Idx, typename G::template Partition<int, 0>::Idx>;
    requires std::same_as<typename G::Idx, typename G::template Field<int, 0>::Idx>;
    
    // All Span types must be identical across the hierarchy
    requires std::same_as<typename G::Span, typename G::template Partition<int, 0>::Span>;
    requires std::same_as<typename G::Span, typename G::template Field<int, 0>::Span>;
    
    // Field partition type must match Grid partition instantiation
    requires std::same_as<typename G::template Partition<int, 0>, typename G::template Field<int, 0>::Partition>;
    
} && requires(G grid) {
    // ═══════════════════════════════════════════════════════════════════════════════════
    // FUNCTIONAL INTERFACE REQUIREMENTS  
    // ═══════════════════════════════════════════════════════════════════════════════════
    
    // Grid must provide span access for partition table mechanism
    {
        grid.getSpan(Neon::Execution::device,
                     Neon::SetIdx(0),
                     Neon::DataView::STANDARD)
    } -> std::same_as<const typename G::Span&>;
    
    // Grid must provide access to span table
    {
        grid.getSpanTable()
    } -> std::convertible_to<const typename G::SpanTable&>;
    
    // Grid must be able to create new fields that satisfy the Field concept
    requires Neon::Field<decltype(grid.template newField<int, 0>("fieldName",
                                                                 int(0),
                                                                 int(0),
                                                                 Neon::DataUse::HOST_DEVICE,
                                                                 Neon::MemoryOptions()))>;
    
    requires Neon::Field<decltype(grid.template newField<double, 1>("fieldName",
                                                                   double(0),
                                                                   double(0),
                                                                   Neon::DataUse::HOST_DEVICE,
                                                                   Neon::MemoryOptions()))>;
};

}  // namespace Neon::domain
