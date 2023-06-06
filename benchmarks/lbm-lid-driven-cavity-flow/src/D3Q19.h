#pragma once

#include "Neon/Neon.h"
#include "Neon/set/Backend.h"
#include "Neon/set/memory/memSet.h"

template <typename StorageFP, typename ComputeFP>
struct D3Q19Template
{
   public:
    static constexpr int Q = 19; /** number of directions */
    static constexpr int D = 3;  /** Space dimension */

    static constexpr int centerDirection = 9; /** Position of direction {0,0,0} */
    static constexpr int goRangeBegin = 0;    /** Symmetry is represented as "go" direction and the "back" their opposite */
    static constexpr int goRangeEnd = 8;
    static constexpr int goBackOffset = 10; /** Offset to compute apply symmetry */


    explicit D3Q19Template(const Neon::Backend& backend)
    {
        // The discrete velocities of the Lattice mesh.
        c_vect = std::vector<Neon::index_3d>(
            {
                {-1, 0, 0} /*!  0  Symmetry first section (GO) */,
                {0, -1, 0} /*!  1  */,
                {0, 0, -1} /*!  2  */,
                {-1, -1, 0} /*! 3  */,
                {-1, 1, 0} /*!  4  */,
                {-1, 0, -1} /*! 5  */,
                {-1, 0, 1} /*!  6  */,
                {0, -1, -1} /*! 7  */,
                {0, -1, 1} /*!  8  */,
                {0, 0, 0} /*!   9  The center */,
                {1, 0, 0} /*!   10 Symmetry mirror section (BK) */,
                {0, 1, 0} /*!   11 */,
                {0, 0, 1} /*!   12 */,
                {1, 1, 0} /*!   13 */,
                {1, -1, 0} /*!  14 */,
                {1, 0, 1} /*!   15 */,
                {1, 0, -1} /*!  16 */,
                {0, 1, 1} /*!   17 */,
                {0, 1, -1} /*!  18 */,
            });

        auto c_neon = backend.devSet().newMemSet<Neon::index_3d>(
            Neon::DataUse::HOST_DEVICE,
            1,
            Neon::MemoryOptions(),
            backend.devSet().newDataSet<size_t>([&](Neon::SetIdx const&, auto& val) {
                val = c_vect.size();
            }));

        for (Neon::SetIdx i = 0; i < backend.devSet().setCardinality(); i++) {
            for (int j = 0; j < int(c_vect.size()); j++) {
                c_neon.eRef(i, j).x = static_cast<int8_t>(c_vect[j].x);
                c_neon.eRef(i, j).y = static_cast<int8_t>(c_vect[j].y);
                c_neon.eRef(i, j).z = static_cast<int8_t>(c_vect[j].z);
            }
        }
        // The opposite of a given direction.
        std::vector<int> opp_vect = {
            10 /*!  0   */,
            11 /*! 1  */,
            12 /*! 2  */,
            13 /*! 3  */,
            14 /*! 4  */,
            15 /*! 5  */,
            16 /*! 6  */,
            17 /*! 7  */,
            18 /*! 8  */,
            9 /*!  9 */,
            0 /*!  10  */,
            1 /*!  11 */,
            2 /*!  12 */,
            3 /*!  13 */,
            4 /*!  14 */,
            5 /*!  15 */,
            6 /*!  16 */,
            7 /*!  17 */,
            8 /*!  18 */,
        };

        {  // Check correctness of opposite
            for (int i = 0; i < static_cast<int>(c_vect.size()); i++) {
                auto point = c_vect[i];
                auto opposite = point * -1;
                if (opposite != c_vect[opp_vect[i]]) {
                    Neon::NeonException exp("");
                    exp << "Incompatible opposite";
                    NEON_THROW(exp);
                }
            }
        }

        this->opp = backend.devSet().newMemSet<int>(
            Neon::DataUse::HOST_DEVICE,
            1,
            Neon::MemoryOptions(),
            backend.devSet().newDataSet<size_t>([&](Neon::SetIdx const&, auto& val) {
                val = opp_vect.size();
            }));


        for (Neon::SetIdx i = 0; i < backend.devSet().setCardinality(); i++) {
            for (size_t j = 0; j < opp_vect.size(); j++) {
                this->opp.eRef(i, j, 0) = opp_vect[j];
            }
        }

        // The lattice weights.
        t_vect = {
            1. / 18. /*!  0   */,
            1. / 18. /*!  1   */,
            1. / 18. /*!  2   */,
            1. / 36. /*!  3   */,
            1. / 36. /*!  4   */,
            1. / 36. /*!  5   */,
            1. / 36. /*!  6   */,
            1. / 36. /*!  7   */,
            1. / 36. /*!  8   */,
            1. / 3. /*!   9  */,
            1. / 18. /*!  10   */,
            1. / 18. /*!  11  */,
            1. / 18. /*!  12  */,
            1. / 36. /*!  13  */,
            1. / 36. /*!  14  */,
            1. / 36. /*!  15  */,
            1. / 36. /*!  16  */,
            1. / 36. /*!  17  */,
            1. / 36. /*!  18  */,
        };

        this->t = backend.devSet().newMemSet<StorageFP>(
            Neon::DataUse::HOST_DEVICE,
            1,
            Neon::MemoryOptions(),
            backend.devSet().newDataSet<size_t>([&](Neon::SetIdx const&, auto&val) {
                val= opp_vect.size();
            }));


        for (Neon::SetIdx i = 0; i < backend.devSet().setCardinality(); i++) {
            for (size_t j = 0; j < t_vect.size(); j++) {
                this->t.eRef(i, j, 0) = t_vect[j];
            }
        }

        if (backend.runtime() == Neon::Runtime::stream) {
            this->c.template update<Neon::run_et::et::sync>(backend.streamSet(0), Neon::DeviceType::CUDA);
            this->opp.template update<Neon::run_et::et::sync>(backend.streamSet(0), Neon::DeviceType::CUDA);
            this->t.template update<Neon::run_et::et::sync>(backend.streamSet(0), Neon::DeviceType::CUDA);
        }
    }


    template <int go>
    static constexpr auto getOpposite()
        -> int
    {
        if constexpr (go == centerDirection)
            return centerDirection;
        if constexpr (go <= goRangeEnd)
            return go + goBackOffset;
        if constexpr (go <= goRangeEnd + goBackOffset)
            return go - goBackOffset;
    }


    Neon::set::MemSet<Neon::int8_3d> c;
    Neon::set::MemSet<int>           opp;
    Neon::set::MemSet<StorageFP>     t;
    std::vector<double>              t_vect;
    std::vector<Neon::index_3d>      c_vect;
};
