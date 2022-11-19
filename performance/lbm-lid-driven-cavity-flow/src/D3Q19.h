#include "Neon/Neon.h"
#include "Neon/set/Backend.h"
#include "Neon/set/memory/memSet.h"
#include "Neon/set/memory/memory.h"

template <typename StorageFP, typename ComputeFP>
struct D3Q19
{
   public:
    D3Q19(const Neon::Backend& bc);

    Neon::set::MemSet_t<Neon::index_3d> c;
    Neon::set::MemSet_t<int>            opp;
    Neon::set::MemSet_t<StorageFP>      t;
};

template <typename StorageFP, typename ComputeFP>
D3Q19<StorageFP, ComputeFP>::D3Q19(const Neon::Backend& backend)
{
    // The discrete velocities of the D3Q19 mesh.
    auto points = std::vector<Neon::index_3d>(
        {{-1, 0, 0} /*!  0  */,
         {0, -1, 0} /*!  1  */,
         {0, 0, -1} /*!  2  */,
         {-1, -1, 0} /*! 3  */,
         {-1, 1, 0} /*!  4  */,
         {-1, 0, -1} /*! 5  */,
         {-1, 0, 1} /*!  6  */,
         {0, -1, -1} /*! 7  */,
         {0, -1, 1} /*!  8  */,
         {1, 0, 0} /*!   9  */,
         {0, 1, 0} /*!   10 */,
         {0, 0, 1} /*!   11 */,
         {1, 1, 0} /*!   12 */,
         {1, -1, 0} /*!  13 */,
         {1, 0, 1} /*!   14 */,
         {1, 0, -1} /*!  15 */,
         {0, 1, 1} /*!   16 */,
         {0, 1, -1} /*!  17 */,
         {0, 0, 0} /*!   Zero is the last one!!! */});

    auto c_neon = Neon::set::Memory::MemSet<Neon::index_3d>(backend, 1, points.size(),
                                                            Neon::DataUse::IO_COMPUTE);


    for (Neon::SetIdx i = 0; i < backend.devSet().setCardinality(); i++) {
        for (size_t j = 0; j < points.size(); j++) {
            c_neon.eRef(i, j) = points[j];
        }
    }
    // The opposite of a given direction.
    std::vector<int> opp_vect = {9 /*! 0   */,
                                 10 /*! 1  */,
                                 11 /*! 2  */,
                                 12 /*! 3  */,
                                 13 /*! 4  */,
                                 14 /*! 5  */,
                                 15 /*! 6  */,
                                 16 /*! 7  */,
                                 17 /*! 8  */,
                                 0 /*!  9  */,
                                 1 /*!  10 */,
                                 2 /*!  11 */,
                                 3 /*!  12 */,
                                 4 /*!  13 */,
                                 5 /*!  14 */,
                                 6 /*!  15 */,
                                 7 /*!  16 */,
                                 8 /*!  17 */,
                                 18 /*! 18 */};

    {  // Check correctness of opposite
        for (int i = 0; i < static_cast<int>(points.size()); i++) {
            auto point = points[i];
            auto opposite = point * -1;
            if (opposite != points[opp_vect[i]]) {
                Neon::NeonException exp("");
                exp << "Incompatible opposite";
                NEON_THROW(exp);
            }
        }
    }

    this->opp = Neon::set::Memory::MemSet<int>(backend, 1, opp_vect.size(),
                                               Neon::DataUse::IO_COMPUTE);

    for (Neon::SetIdx i = 0; i < backend.devSet().setCardinality(); i++) {
        for (size_t j = 0; j < opp_vect.size(); j++) {
            this->opp.eRef(i, j, 0) = opp_vect[j];
        }
    }

    // The lattice weights.
    std::vector<double> t_vect = {
        1. / 18. /*!  0   */,
        1. / 18. /*!  1   */,
        1. / 18. /*!  2   */,
        1. / 36. /*!  3   */,
        1. / 36. /*!  4   */,
        1. / 36. /*!  5   */,
        1. / 36. /*!  6   */,
        1. / 36. /*!  7   */,
        1. / 36. /*!  8   */,
        1. / 18. /*!  9   */,
        1. / 18. /*!  10  */,
        1. / 18. /*!  11  */,
        1. / 36. /*!  12  */,
        1. / 36. /*!  13  */,
        1. / 36. /*!  14  */,
        1. / 36. /*!  15  */,
        1. / 36. /*!  16  */,
        1. / 36. /*!  17  */,
        1. / 3. /*!   18  */,
    };

    this->t = Neon::set::Memory::MemSet<double>(backend, 1, opp_vect.size(),
                                                Neon::DataUse::IO_COMPUTE);

    for (Neon::SetIdx i = 0; i < backend.devSet().setCardinality(); i++) {
        for (size_t j = 0; j < t_vect.size(); j++) {
            this->t.eRef(i, j, 0) = t_vect[j];
        }
    }

    if (backend.runtime() == Neon::Runtime::stream) {
        this->c.update<Neon::run_et::et::sync>(backend.streamSet(0), Neon::DeviceType::CUDA);
        this->c.opp.update<Neon::run_et::et::sync>(backend.streamSet(0), Neon::DeviceType::CUDA);
        this->t.update<Neon::run_et::et::sync>(backend.streamSet(0), Neon::DeviceType::CUDA);
    }
}