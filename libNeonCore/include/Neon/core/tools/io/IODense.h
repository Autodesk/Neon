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
#include "Neon/core/types/vec.h"

namespace Neon {

/**
 * An abstraction for fields over a dense 3D grid.
 * @tparam ExportType
 * @tparam IntType
 */
template <typename ExportType, typename IntType = int>
struct IODense
{
   private:
    enum struct Representation
    {
        IMPLICIT,
        EXPLICIT
    };

   public:
    using Index = IntType /** type used to index elements */;
    using Type = ExportType /** Type of the data stored in the grid cells */;

    IODense();

    IODense(const Integer_3d<IntType>&     d,
            int                            c,
            std::shared_ptr<ExportType[]>& m,
            Neon::memLayout_et::order_e    order = Neon::memLayout_et::order_e::structOfArrays);

    IODense(const Integer_3d<IntType>&  d,
            int                         c,
            ExportType*                 m,
            Neon::memLayout_et::order_e order = Neon::memLayout_et::order_e::structOfArrays);

    IODense(const Integer_3d<IntType>&  d,
            int                         c,
            Neon::memLayout_et::order_e order = Neon::memLayout_et::order_e::structOfArrays);

    IODense(const Integer_3d<IntType>&                                              d,
            int                                                                     c,
            const std::function<ExportType(const Integer_3d<IntType>&, int cardinality)>& fun);
    /**
     * Generate a dense field using an implicit representation.
     */
    template <typename Lambda_ta>
    static auto densify(const Lambda_ta&           fun /*!            Implicit definition of the user field */,
                        const Integer_3d<IntType>& space /*!          dense grid dimension */,
                        int                        cardinality /*!    Field cardinality */,
                        Representation             representation = Representation::EXPLICIT)
        -> IODense<ExportType, IntType>;

    /**
     * Initializes the dense buffer linearly following lexicographic order.
     */
    static auto makeLinear(ExportType                 offset,
                           const Integer_3d<IntType>& space /*!          dense grid dimension */,
                           int                        cardinality /*!    Field cardinality */)
        -> IODense<ExportType>;

    /**
     * Initializes the value with random numbers in the rage [min,max]
     * @return
     */
    static auto makeRandom(int                        min,
                           int                        max,
                           const Integer_3d<IntType>& space /*!          Dense grid dimension */,
                           int                        cardinality /*!    Field cardinality */)
        -> IODense<ExportType>;

    /**
     * Initializes the value with a bit mask computed w.r.t to each element x,y,z position
     * This is useful initialization for debugging.
     *
     * if digit == 3 and (x,y,z) == (1,2,5) -> 1002005
     */
    static auto makeMasked(ExportType                 offset,
                           const Integer_3d<IntType>& space /*!          Dense grid dimension */,
                           int                        cardinality /*!    Field cardinality */,
                           int                        digit = 2)
        -> IODense<ExportType>;

    auto resetValuesToLinear(ExportType offset)
        -> void;

    /**
     * Initializes the value with random numbers in the rage [min,max]
     * @return
     */
    auto resetValuesToRandom(int min,
                             int max)
        -> void;

    /**
     * Initializes the value with a bit mask computed w.r.t to each element x,y,z position
     * This is useful initialization for debugging.
     *
     * if digit == 3 and (x,y,z) == (1,2,5) -> 1002005
     */
    auto resetValuesToMasked(ExportType offset,
                             int        digit = 2)
        -> void;

    auto resetValuesToConst(ExportType offset)
        -> void;

    /**
     * The space of the gris
     */
    auto getDimension() const
        -> Integer_3d<IntType>;

    /**
     * Cardinality of the data over the grid
     * @return
     */
    auto getCardinality() const -> int;

    auto getMemory() -> ExportType*;

    auto getSharedPtr() -> std::shared_ptr<ExportType[]>;

    /**
     * Accessing a point in the field. Read only mode.
     */
    auto getReference(const Integer_3d<IntType>& xyz /**< Point in the grid        */,
                      int                        card /**< Cardinality of the field */)
        -> ExportType&;

    auto operator()(const Integer_3d<IntType>& xyz /**< Point in the grid        */,
                    int                        card /**< Cardinality of the field */) const
        -> ExportType;
    /**
     * For each operator to visit all field elements in parallel
     */
    template <typename Lambda_ta, typename... ExportTypeVariadic_ta>
    auto forEach(const Lambda_ta& lambda /**< User function                                                    */,
                 IODense<ExportTypeVariadic_ta>&... otherDense /**< Optional. Other fields that may be needed during the field visit */)
        -> void;

    /**
     * For each operator to visit all field elements in parallel.
     * Read only mode
     */
    template <typename Lambda_ta, typename... ExportTypeVariadic_ta>
    auto forEach(const Lambda_ta& lambda /**< User function                                                    */,
                 const IODense<ExportTypeVariadic_ta>&... otherDense /**< Optional. Other fields that may be needed during the field visit */)
        const -> void;

    /**
     * Computing the max different component by component.
     */
    static auto maxDiff(const IODense<ExportType, IntType>& a,
                        const IODense<ExportType, IntType>& b)
        -> std::tuple<ExportType /**< the max difference value */,
                      Neon::index_3d /**< the location of the max difference */,
                      int /**< the cardinality of the max difference */>;

    /**
     * Exporting to vtk
     * @return
     */
    template <typename ExportTypeVTK_ta = ExportType>
    auto ioVtk(const std::string&       filename /*!                              File name */,
               const std::string&       fieldName /*!                              Field name */,
               ioToVTKns::VtiDataType_e nodeOrVoxel = ioToVTKns::VtiDataType_e::node,
               const Vec_3d<double>&    spacingData = Vec_3d<double>(1, 1, 1) /*! Spacing, i.e. size of a voxel */,
               const Vec_3d<double>&    origin = Vec_3d<double>(0, 0, 0) /*!      Origin  */,
               IoFileType               vtiIOe = IoFileType::ASCII /*!            Binary or ASCII file  */)
        -> void;

   private:

    auto initPitch() -> void;

    std::shared_ptr<ExportType[]>                                          mMemSharedPtr;
    ExportType*                                                            mMem;
    Integer_3d<IntType>                                                    mSpace /*! IoDense dimension */;
    int                                                                    mCardinality;
    Neon::memLayout_et::order_e                                            mOrder;
    Neon::size_4d                                                          mPitch;
    Representation                                                         mRepresentation;
    std::function<ExportType(const Integer_3d<IntType>&, int cardinality)> mImplicitFun;
};

}  // namespace Neon

#include "Neon/core/tools/io/IODense_imp.h"