#pragma once

#include <string>

#include "Neon/core/core.h"

#include "Neon/set/Backend.h"
#include "Neon/set/DataSet.h"
#include "Neon/set/DevSet.h"

#include "Neon/core/tools/io/ioToVti.h"
#include "Stencil.h"

namespace Neon::domain::interface {

/**
 * Class containing common tools/operations/data for any grid.
 */
class GridBase
{
   public:
    GridBase();

    /**
     * Returns the size of the grid
     */
    auto getDimension() const
        -> const Neon::index_3d&;

    /**
     * Returns the stencil used for the grid initialization
     */
    auto getStencil() const
        -> const Neon::domain::Stencil&;

    /**
     * Return spacing between cells
     */
    auto getSpacing() const
        -> const Vec_3d<double>&;

    /**
     * Return the origin of the background grid.
     */
    auto getOrigin() const
        -> const Vec_3d<double>&;

    /**
     * Returns total number of cells.
     */
    auto getNumAllCells() const
        -> size_t;

    /**
     * Returns total number of cells.
     */
    auto getNumActiveCells() const
        -> size_t;

    virtual auto isInsideDomain(const Neon::index_3d& idx) const
        -> bool = 0;

    [[deprecated("Will be replace by the getNumActiveCellsPerPartition method")]] auto
    flattenedLengthSet() const
        -> const Neon::set::DataSet<size_t>&;

    /**
     * Return the number of cells stored per partition
     * @return
     */
    auto getNumActiveCellsPerPartition() const
        -> const Neon::set::DataSet<size_t>&;

    /**
     * Creates a DataSet object compatible with the number of GPU used by the grid.
     */
    template <typename T>
    auto newDataSet() const
        -> const Neon::set::DataSet<T>;

    /**
     * Returns the name of the grid (for example eGrid, dGrid..).
     * */
    auto getImplementationName() const
        -> const std::string&;

    /**
     * Returns the launch parameters based on the specified data view mode.
     */
    auto getDefaultLaunchParameters(Neon::DataView) const
        -> const Neon::set::LaunchParameters&;

    /**
     * Returns the backed used to create the grid
     */
    auto getBackend() const
        -> const Backend&;

    /**
     * Returns the backed used to create the grid
     */
    auto getBackend()
        -> Backend&;

    /**
     * Returns the DevSet object used to create the grid.
     */
    auto getDevSet() const
        -> const Neon::set::DevSet&;

    /**
     * Returns a string describing the grid
     */
    auto toString() const
        -> std::string;

    /**
     * Returns an UID for the grid
     * @return
     */
    auto getGridUID() const
        -> size_t;

    /**
     * Add the grid information in a Report object
     */
    virtual auto toReport(Neon::Report& report,
                          bool          addBackendInfo = false) const
        -> void;

    /**
     * Returns the thread block size used by default by the grid
     */
    auto getDefaultBlock() const
        -> const Neon::index_3d&;


   protected:
    /**
     * Protected constructor
     */
    GridBase(const std::string&                gridImplementationName,
             const Neon::Backend&              backend,
             const Neon::index_3d&             dim,
             const Neon::domain::Stencil&      stencil,
             const Neon::set::DataSet<size_t>& nPartitionElements /**< Number of element per partition */,
             const Neon::index_3d&             defaultBlockSize,
             const Vec_3d<double>&             spacingData = Vec_3d<double>(1, 1, 1) /*! Spacing, i.e. size of a voxel */,
             const Vec_3d<double>&             origin = Vec_3d<double>(0, 0, 0) /*!      Origin  */);

    /**
     * Protected initialization function used by derived classes to set some parameters.
     */
    auto init(const std::string&                gridImplementationName /**< Name of the implementation, for example dGrid eGrid etc */,
              const Neon::Backend&              backend /**< Backend used to create the grid */,
              const Neon::index_3d&             dimension /**< Dimension of the grid */,
              const Neon::domain::Stencil&      stencil /**< Union of all the stencil that will be used with the grid */,
              const Neon::set::DataSet<size_t>& nPartitionElements /**< Elements associated to each partition */,
              const Neon::index_3d&             defaultBlockSize /**< Default thread block size */,
              const Vec_3d<double>&             spacingData /**< Grid spacing */,
              const Vec_3d<double>&             origin /**< Position in space of the grid's origin */) -> void;

    /**
     * Protected method to set the default thread blocks size
     */
    auto setDefaultBlock(const Neon::index_3d&)
        -> void;

    /**
     * Protected method to set default thread block size by data view.
     */
    auto getDefaultLaunchParameters(Neon::DataView)
        -> Neon::set::LaunchParameters&;


   private:
    struct Storage
    {
        struct Defaults_t
        {
            std::array<Neon::set::LaunchParameters,
                       Neon::DataViewUtil::nConfig>
                launchParameters;

            index_3d blockDim;
        };

        Neon::Backend              backend /**<            Backend used to create and run the grid. */;
        Neon::index_3d             dimension /**<          Dimension of the grid                    */;
        Neon::domain::Stencil      stencil /**<            Stencil used for the grid initialization */;
        Neon::set::DataSet<size_t> nPartitionElements /**< Number of elements per partition         */;
        Vec_3d<double>             spacing /**<            Spacing, i.e. size of a voxel            */;
        Vec_3d<double>             origin /**<             Position in space of the grid's origin   */;
        Defaults_t                 defaults;
        std::string                gridImplementationName;
    };

    std::shared_ptr<Storage> mStorage;
};

}  // namespace Neon::domain::interface

#include "GridBase_imp.h"
