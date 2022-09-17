#pragma once
#include "Neon/set/Backend.h"
#include "Neon/set/HuOptions.h"
#include "Neon/set/dependency/AccessType.h"
#include "Neon/set/dependency/Alias.h"
#include "Neon/set/dependency/ComputeType.h"

namespace Neon::internal::dataDependency {

/**
 * Stores type of operations on data for each kernels while user code is "parsed"
 * It is used to construct the user kernel dependency graph
 */
struct Token
{
   public:
    Token() = delete;

    /**
     * Unique constructor
     */
    Token(DataUId       m_uid,
          AccessType    m_access,
          Neon::Compute m_compute);

    /**
     * Method to update a token
     */
    auto update(DataUId       m_uid,
                AccessType    m_access,
                Neon::Compute m_compute) -> void;

    /**
     * It returns the multi-GPU data uid
     */
    auto uid() const -> DataUId;

    /**
     * It returns the type of data access
     */
    auto access() const -> AccessType;

    /**
     * Returns the compute pattern
     */
    auto compute() const -> Neon::Compute;

    /**
     * Converts the token into a string
     */
    auto toString() const -> std::string;

    /**
     * It sets the halo update container associated to the multi-GPU data.
     * @param hu
     * @param huPerDevice
     */
    auto setHaloUpdate(std::function<void(Neon::set::HuOptions& opt)>               hu,
                       std::function<void(Neon::SetIdx, Neon::set::HuOptions& opt)> huPerDevice) -> void;

    auto getHaloUpdate() const -> const std::function<void(Neon::set::HuOptions& opt)>&;

    auto getHaloUpdatePerDevice() const -> const std::function<void(Neon::SetIdx, Neon::set::HuOptions& opt)>&;

    auto mergeAccess(AccessType) -> void;


   private:
    DataUId                                                      mUid;
    AccessType                                                   mAccess;
    Compute                                                      mCompute;
    std::function<void(Neon::set::HuOptions& opt)>               mHu;
    std::function<void(Neon::SetIdx, Neon::set::HuOptions& opt)> mHuPerDevice;
};

}  // namespace dependency
