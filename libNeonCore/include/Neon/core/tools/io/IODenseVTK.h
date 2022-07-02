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
#include "Neon/core/tools/io/IODense.h"
#include "Neon/core/tools/io/ioToVTK.h"

namespace Neon {

/**
 * Structure to export a dense field to a VTK file.
 * Multiple fields can be stored int the same VTK file
 */
template <typename intType_ta = int, class real_tt = double>
class IODenseVTK : private IoToVTK<intType_ta, real_tt>
{

   public:
    IODenseVTK(const std::string&           filename /*! File name */,
               const Vec_3d<double>&        spacingData = Vec_3d<double>(1, 1, 1) /*! Spacing, i.e. size of a voxel */,
               const Vec_3d<double>&        origin = Vec_3d<double>(0, 0, 0) /*! Origin  */,
               ioVTI_e::e                   vtiIOe = ioVTI_e::e::ASCII /*!  Binary or ASCII file  */);

    /**
     * Add a field to the file
     */
    template <typename ExportType_ta>
    auto addField(IODense<ExportType_ta, intType_ta> dense,
                  const std::string&                 fname /*!   Name of the field */,
                  bool                               isNodeSpace) -> void;

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
    Neon::Integer_3d<intType_ta> m_nodeSpace;
};

}  // namespace Neon

#include "Neon/core/tools/io/IODenseVTK_imp.h"