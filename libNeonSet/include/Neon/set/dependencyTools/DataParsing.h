#pragma once
#include "Neon/set/Backend.h"
#include "Neon/set/HuOptions.h"
#include "Neon/set/dependencyTools/Alias.h"
#include "Neon/set/dependencyTools/enum.h"

namespace Neon {
namespace set {
namespace internal {
namespace dependencyTools {

/**
 * Stores type of operations on data for each kernels while user code is "parsed"
 * It is used to construct the user kernel dependency graph
 */
struct DataToken
{
   private:
    DataUId_t                                                    m_uid;
    Access_e                                                     m_access;
    Compute                                                      m_compute;
    std::function<void(Neon::set::HuOptions& opt)>               m_hu;
    std::function<void(Neon::SetIdx, Neon::set::HuOptions& opt)> m_huPerDevice;

   public:
    DataToken() = delete;

    DataToken(DataUId_t     m_uid,
              Access_et::e  m_access,
              Neon::Compute m_compute);

    auto update(DataUId_t     m_uid,
                Access_et::e  m_access,
                Neon::Compute m_compute) -> void;
    auto uid() const -> DataUId_t;
    auto access() const -> Access_et::e;
    auto compute() const -> Neon::Compute;
    auto toString() const -> std::string;

    auto setHaloUpdate(std::function<void(Neon::set::HuOptions& opt)>               hu,
                       std::function<void(Neon::SetIdx, Neon::set::HuOptions& opt)> huPerDevice) -> void;

    auto getHaloUpdate() const -> const std::function<void(Neon::set::HuOptions& opt)>&;
    auto getHaloUpdatePerDevice() const -> const std::function<void(Neon::SetIdx, Neon::set::HuOptions& opt)>&;

    auto mergeAccess( Access_et::e)->void ;
};

}  // namespace dependencyTools
}  // namespace internal
}  // namespace set
}  // namespace Neon