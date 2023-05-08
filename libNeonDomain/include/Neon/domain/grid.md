# Neon Grid Abstractions

Neon Domain library provide an abstraction to manage free form domain that are discredited on Euclidean domain. Grids in
Neon allow user to define a domain, associate metadata to the discretisation points and to run containers on top of
them. Neon domain automatically manges the distribution of the discretization points over the available resources.

LibNeonGrid include a set of different grids: from dense to sparse and in short future also multi-resolution domain. All
the grids share the same user API so that user can easilly switch from one implementation to another. As for now, Neon
does not provide any mechanism to switch between domain at runtime; the operation is left to the user design.

## Provided grids

- `aGrid`: represents a 1D grid. Its main use is as support for other domain.
- `dGrid`: is a dense representation of a domain. It stores in memory both active and non-active elements.
- `eGrid`: is an element sparse representation. Only active elements are represented. To support stencil operations,
  eGrid relays on an explicit connectivity table

## Sub grid abstraction

User may need to store data and run some computation only on a sub set of a grid cells. To help in the process, Neon
provide sGrid. An sGrid can be created from `dGrid` and `eGrid`.

## How to implement a new grid

Neon Domain level can be extended with user defined grids. The following are the required steps to implement a new grid.
For simplicity will refer as xGrid as the new grid. The classes should implement the following classes:

- `xCell`: abstraction to define a cell handler.
- `xPartitionIndexSpace`: abstraction for the index space of the computation
- `xPartition`: abstraction for the actual data stored by the Fields. This is the object user will leverage in their
  Containers.
- `xGrid`: grid abstraction
- `xField`: field abstraction that manages user metada on the gris. The Field manages the different partitions where the
  metadata is stored.

We suggest following this exact order to avoid any issue compilation with cyclic dependencies. Neon uses abstract class
and templates to define the interface of a new grid. Abstract class mechanism are used to reduce replicated code.
Performance critical API are based on a template interface. In the future,  `concepts` will be used to describe such
interface (we are waiting for `concepts` to be supported in CUDA).

### Step 0 - Define a Idx type

First goal is to define an abstraction for a cell handler. The cell handler is provided by the system to the user to
access cell metadata.

```c++
struct xCell
{
    using OuterIndex = xCell;

    friend struct sPartitionIndexSpace;

    template <typename T,
              int Cardinality>
    friend struct xPartition;

    friend class xGrid;

    xCell() = default;

private:
    //..
};

```

The definition of `OuterIndex` is required to be able to use the sGrid to define sub-grids. The most easy is to define
the `OuterIndex` with the same type of the xCell, however, depending on the type of grid this may not be the most
efficient way.

### Step 1 - Design a xPartitionIndexSpace

xPartitionIndexSpace is the abstraction for the index space of the computation, the "thread grid" in CUDA jargon.

```c++
class xPartitionIndexSpace {
 public:
    friend class xGrid;

    using Idx = xCell;
    static constexpr int SpaceDim = 1;

    NEON_CUDA_HOST_DEVICE
    inline auto setAndValidate(Idx&                          cell,
                               const size_t&                  x,
                               [[maybe_unused]] const size_t& y,
                               [[maybe_unused]] const size_t& z)const
        -> bool;

   private:
    //...
}
```

The class must stratically specify the dimention of the index space (1D, 2D or 3D)  thorough the static
integer `SpaceDim`. The only exposed methods is `setAndValidate`, which is used by the runtime to set a Idx handler
from the position of the running thread w.r.t. the thread grid.

### Step 2 - Creating a xPartition class

xPartition is the abstraction of a setIdx. It manages the field metadata on a specific device.

```c++
template <typename T /**< Type of the element of the setIdx */,
          int C = 0 /** Cardinality of the field. If zero, the cardinality is determined at runtime */>
class xPartition
{
   public:
    // These types must be declared. 
    // The actual type they represent depends on the grid
    using Type = T;
    using Idx = int64_t;

   public:

    /**
     * Returns the setIdx index.
     */
    NEON_CUDA_HOST_DEVICE auto 
    partitionIdx() const
        -> const Neon::index3d&;
    
    /**
     * Returns the metadata associated with cell referred by eIdx.
     * This method should be used only for fields of cardinality 1
     */
    NEON_CUDA_HOST_DEVICE inline auto 
    operator()(const Idx& eId, int cardinalityIdx) const
        -> const T&;

    /**
     * Returns the metadata associated with cell referred by eIdx.
     * This method should be used only for fields of cardinality 1
     */
    NEON_CUDA_HOST_DEVICE inline auto 
    operator()(const Idx& eId, int cardinalityIdx) 
        ->  T&;
    
    /**
     * Returns the metadata associated with a neighbour cell.
     */
    NEON_CUDA_HOST_DEVICE inline auto 
    getNghData(const Idx& eId, int stencilPointIdx, int cardinalityIdx, T& value) const
        -> bool;
    
    /**
     * Returns the metadata associated with a neighbour cell.
     */
    NEON_CUDA_HOST_DEVICE inline auto 
    getNghData(Idx eId, const Neon::int3d& direrection, int cardinalityIdx, T& value) const
        -> bool;
    
    /**
     * Returns the field cardinality
     */
    NEON_CUDA_HOST_DEVICE inline auto 
    cardinality() const
        -> int;
    
    NEON_CUDA_HOST_DEVICE inline auto 
    auto getLocation() const
    -> Neon::index_3d;
    
private:
//...
```

### Step 3 - Creating a xField class

xField is the abstraction for metadata associated to the grid cells: the field. The xField extends
the `FieldBaseTemplate` class that requires information on both the xGrid and xPartition types.

```c++
#include "Neon/domain/interface/FieldBaseTemplate.h"
  
    friend sGrid<OuterGridT>;

    // New Naming:
    using Partition = sPartition<OuterGridT, T, C>; /**< Type of the associated fieldCompute */
    using Type = typename Partition::Type /**< Type of the information stored in one element */;
    using Idx = typename Partition::Idx /**< Internal type that represent the location in memory of a element */;
    using Grid = sGrid<OuterGridT>;

    static constexpr int Cardinality = C;

    // ALIAS
    using Self = sField<OuterGridT, Type, Cardinality>;

    using Count = typename Partition::Count;
    using Index = typename Partition::Index;


   public:
    sField();

    auto self() -> Self&;

    auto self() const -> const Self&;

    /**
     * Returns the metadata associated with the element in location idx.
     * If the element is not active (it does not belong to the voxelized domain),
     * then the default outside value is returned.
     */
    auto operator()(const Neon::index_3d& idx,
                    const int&            cardinality) const
        -> Type final;

    virtual auto getReference(const Neon::index_3d& idx,
                              const int&            cardinality)
        -> Type& final;

    auto haloUpdate(Neon::set::HuOptions& opt) const
        -> void final;

    auto haloUpdate(Neon::set::HuOptions& opt)
        -> void final;

    /**
     * Move the field metadata from host to the accelerators.
     * The operation is asynchronous.
     */
    auto updateDeviceData(int streamIdx = 0)
        -> void;

    /**
     * Move the field metadata from the accelerators to the host space.
     * The operation is asynchronous.
     */
    auto updateHostData(int streamIdx = 0)
        -> void;

    [[deprecated("Will be replace by the getPartition method")]] auto
    getPartition(Neon::DeviceType      devEt,
                 Neon::SetIdx          setIdx,
                 const Neon::DataView& dataView = Neon::DataView::STANDARD) const -> const Partition&;

    [[deprecated("Will be replace by the getPartition method")]] auto
    getPartition(Neon::DeviceType      devEt,
                 Neon::SetIdx          setIdx,
                 const Neon::DataView& dataView = Neon::DataView::STANDARD) -> Partition&;

    /**
     * Return a constant reference to a specific setIdx based on a set of parameters:
     * execution type, target device, dataView
     */
    auto getPartition(Neon::Execution       execution,
                      Neon::SetIdx          setIdx,
                      const Neon::DataView& dataView = Neon::DataView::STANDARD) const -> const Partition&;
    /**
     * Return a reference to a specific setIdx based on a set of parameters:
     * execution type, target device, dataView
     */
    auto getPartition(Neon::Execution       execution,
                      Neon::SetIdx          setIdx,
                      const Neon::DataView& dataView = Neon::DataView::STANDARD) -> Partition&;
//..
};
```

### Step 3 - Creating a xGrid class

Finally, xGrid is the last piece of the puzzle. The xGrid class publicly derives `Neon::domain::interface::GridBase` and define a set of class types. 
```c++
#include "Neon/set/MemoryOptions.h"

#include "Neon/domain/interface/GridBase.h"

//...
// Forward declaration of xField
template <typename T, int C>
class xField;

class xGrid : public Neon::domain::interface::GridBase
{
   public:

    template <typename T, int C>
    using Partition = xPartition<T, C>; /**< Type of the associated fieldCompute */

    template <typename T, int C = 0>
    using Field = xField<T, C>; 
    using Span = xPartitionIndexSpace;
    //...


    /**
     * Default constructor
     */
    xGrid();

    /**
     * Constructor compatible with the general grid API
     */
    template <typename SparsityPattern>
    xGrid(const Neon::set::Backend& backend,
          const Neon::int32_3d&       dimension /**< Dimension of the box containing the sparse domain */,
          const SparsityPattern      activeCellLambda /**< InOrOutLambda({x,y,z}->{true, false}) */
          const Neon::domain::Stencil&                     stencil,
          const Vec_3d<double>&       spacingData = Vec_3d<double>(1, 1, 1) /**< Spacing, i.e. size of a voxel */,
          const Vec_3d<double>&       origin = Vec_3d<double>(0, 0, 0) /**<      Origin  */);
    
    /**
     * Returns a LaunchParameters configured for the specified inputs
     */
    auto getLaunchParameters(Neon::DataView        dataView,
                             const Neon::index_3d& blockSize = Neon::index_3d(256, 1, 1),
                             size_t                shareMem = 0) const
        -> Neon::set::LaunchParameters;

    /**
     * Returns the setIdx space that can be used by the lambda executor to run a Container
     */
    auto getSpan(Neon::DeviceType devE,
                           SetIdx          setIdx,
                           Neon::DataView   dataView) const
        -> const xGrid::Span&;

    /**
     * Creates a new Field
     */
    template <typename T, int C = 0>
    auto newField(const std::string               fieldUserName,
                  int                             cardinality,
                  T                               inactiveValue,
                  Neon::DataUse                   dataUse = Neon::DataUse::HOST_DEVICE,
                  const Neon::MemoryOptions& memoryOptions = Neon::MemoryOptions()) const
        -> Field<T, C>;

    /**
     * Creates a container
     */
    template <typename LoadingLambda>
    auto getContainer(const std::string& name,
                      LoadingLambda      lambda)
        const
        -> Neon::set::Container;

    /**
     * Creates a container with the ability of specifying the block and shared memory size
     */
    template <typename LoadingLambda>
    auto getContainer(const std::string& name,
                      LoadingLambda      lambda
                      index_3d           blockSize,
                      size_t             sharedMem)
        const
        -> Neon::set::Container;

        template <typename T>
    auto newPatternScalar() const
        -> Neon::template PatternScalar<T>;

    template <typename T, int C>
    auto dot(const std::string& /*name*/,
             eField<T, C>& /*input1*/,
             eField<T, C>& /*input2*/,
             Neon::template PatternScalar<T>& /*scalar*/) const -> Neon::set::Container;

    template <typename T, int C>
    auto norm2(const std::string& /*name*/,
               eField<T, C>& /*input*/,
               Neon::template PatternScalar<T>& /*scalar*/) const -> Neon::set::Container;    

    /**
     * Function that converts a set of points of a stencil from its 3D offset form to a grid internal representation.
     * @param stencilOffsets
     * @return
     */
    auto convertToNgh(const std::vector<Neon::index_3d>& stencilOffsets)
        -> std::vector<ngh_idx>;

    /**
     * Function that converts a point of a stencil from its 3D offset form to a grid internal representation.
     * @param stencilOffsets
     * @return
     */
    auto convertToNgh(const Neon::index_3d& stencilOffset)
        -> Self::ngh_idx;
   private:
//...
```