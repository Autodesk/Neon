/**
 * This file contains a set of tools to manage dense grids on the CPU
 * These are useful to convert any grid into a dense representation,
 * to easily convert external data to a format that Neon grids can load
 * and store. The dense representation also includes capabilities to compare
 * the values of two different grids.
 */
#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>

#include "Neon/core/core.h"
#include "Neon/core/tools/io/ioToVTK.h"

namespace Neon::domain {

/**
 * Structure to export a dense field to a VTK file.
 * Multiple fields can be stored int the same VTK file
 */
template <class real_tt = double, typename intType_ta = int>
class IOGridVTK : private IoToVTK<intType_ta, real_tt>
{

   public:
    IOGridVTK(const Neon::domain::interface::GridBase& grid,
              const std::string&                       filename /*!   File name */,
              bool                                     isNodeSpace = true,
              IoFileType                               vtiIOe = IoFileType::ASCII /*! Binary or ASCII file  */);

    IOGridVTK(const Neon::domain::interface::GridBase& grid,
              const Neon::double_3d&                   oringOffset,
              const std::string&                       filename /*!   File name */,
              bool                                     isNodeSpace = true,
              IoFileType                               vtiIOe = IoFileType::ASCII /*! Binary or ASCII file  */);
    /**
     * Add a field to the file
     */
    template <typename Field>
    auto addField(const Field&       field,
                  const std::string& name /*! Name of the field */) -> void;

    /**
     * Write the VTK file
     */
    using IoToVTK<intType_ta, real_tt>::flush;

    /**
     * Clear all fields already added
     */
    using IoToVTK<intType_ta, real_tt>::clear;

    /**
     * Write the VTK file and clear all fields already added
     */
    using IoToVTK<intType_ta, real_tt>::flushAndClear;

    using IoToVTK<intType_ta, real_tt>::setIteration;

    using IoToVTK<intType_ta, real_tt>::setFormat;

   private:
    ioToVTKns::VtiDataType_e mVtiDataTypeE /**< tells if the space in m_ioToDense is the node or voxel space */;
    Neon::index_3d           mDimension;
};

}  // namespace Neon::domain

#include "Neon/domain/tools/IOGridVTK_imp.h"