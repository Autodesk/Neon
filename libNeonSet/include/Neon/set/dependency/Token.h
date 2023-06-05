#pragma once

#include "Neon/set/Backend.h"
#include "Neon/set/HuOptions.h"
#include "Neon/set/MultiXpuDataUid.h"
#include "Neon/set/dependency/AccessType.h"
#include "Pattern.h"

namespace Neon::set {
struct Container;
}

namespace Neon::set::dataDependency {

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
    Token(Neon::set::dataDependency::MultiXpuDataUid m_uid,
          Neon::set::dataDependency::AccessType      m_access,
          Neon::Pattern                              m_compute);

    /**
     * Method to update a token
     */
    auto update(Neon::set::dataDependency::MultiXpuDataUid m_uid,
                Neon::set::dataDependency::AccessType      m_access,
                Neon::Pattern                              m_compute)
        -> void;

    /**
     * It returns the multi-GPU data uid
     */
    auto uid()
        const -> Neon::set::dataDependency::MultiXpuDataUid;

    /**
     * It returns the type of data access
     */
    auto access()
        const -> Neon::set::dataDependency::AccessType;

    /**
     * Returns the compute pattern
     */
    auto compute()
        const -> Neon::Pattern;

    /**
     * Converts the token into a string
     */
    auto toString()
        const -> std::string;

    /**
     * It sets the halo update container associated to the multi-GPU data.
     */
    auto setDataTransferContainer(std::function<Neon::set::Container(Neon::set::TransferMode transferMode)> huPerDevice)
        -> void;

    auto getDataTransferContainer(Neon::set::TransferMode transferMode)
        const -> Neon::set::Container;

    auto mergeAccess(AccessType)
        -> void;


   private:
    Neon::set::dataDependency::MultiXpuDataUid mUid;
    Neon::set::dataDependency::AccessType      mAccess;
    Neon::Pattern                              mCompute;

    std::function<Neon::set::Container(Neon::set::TransferMode transferMode)> mHaloUpdateExtractor;
};

}  // namespace Neon::set::dataDependency
