#include "Neon/domain/details/bGrid/BlockView/BlockViewGrid.h"
#include "Neon/domain/tools/GridTransformer.h"

namespace Neon::domain::details::bGrid {

struct BlockView
{
   public:
    using Grid = Neon::domain::tool::GridTransformer<details::GridTransformation>::Grid;
    template <typename T, int C = 0>
    using Field = Grid::template Field<T, C>;
    using index_3d = Neon::index_3d;

    template <typename T, int C = 0>
    static auto helpGetReference(T* mem, const int idx, const int card) -> std::enable_if_t<C == 0, T&>
    {
        return mem[idx * card];
    }

    template <typename T, int C = 0>
    static auto helpGetReference(T* mem, const int idx, const int card) -> std::enable_if_t<C != 0, T&>
    {
        return mem[idx * C];
    }

    static constexpr Neon::MemoryLayout layout = Neon::MemoryLayout::arrayOfStructs;
};

}  // namespace Neon::domain::details::bGrid