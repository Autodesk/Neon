#include "Neon/set/Backend.h"

namespace Neon {
namespace distributed {
struct Backend
{
    struct Data;

    Backend();
    ~Backend();


    std::shared_ptr<Data> mData;
};
}  // namespace distributed
}  // namespace neon