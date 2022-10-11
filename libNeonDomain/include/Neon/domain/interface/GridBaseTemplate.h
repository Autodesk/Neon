#pragma once

#include <string>

#include "Neon/core/core.h"

#include "Neon/set/Backend.h"
#include "Neon/set/DataSet.h"
#include "Neon/set/DevSet.h"

#include "Neon/domain/interface/CellProperties.h"
#include "Neon/domain/interface/GridBase.h"

namespace Neon::domain::interface {

template <typename GridT, typename CellT>
class GridBaseTemplate : public GridBase
{
    // https://stackoverflow.com/questions/28002/regular-cast-vs-static-cast-vs-dynamic-cast
    // https://en.wikibooks.org/wiki/C%2B%2B_Programming/Programming_Languages/C%2B%2B/Code/Statements/Variables/Type_Casting
   public:
    using Grid = GridT;
    using Cell = CellT;
    using CellProperties = Neon::domain::interface::CellProperties<Cell>;

    virtual auto getProperties(const Neon::index_3d& idx) const
        -> CellProperties = 0;

    /**
     * Exporting the domain active voxel to vtk
     */
    auto ioDomainToVtk(const std::string& fileName,
                       Neon::IoFileType   vtiIOe = IoFileType::ASCII) const -> void;
};
}  // namespace Neon::domain::interface

#include "Neon/domain/interface/GridBaseTemplate_imp.h"