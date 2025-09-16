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
 * @brief Token representing data dependencies between kernels
 * 
 * Stores information about data operations performed by kernels during the parsing
 * phase of user code. These tokens are used to construct the dependency graph that
 * determines the execution order of kernels in multi-GPU environments.
 * 
 * Each token captures:
 * - The data being accessed (via unique identifier)
 * - The type of access (read, write, read-write)
 * - The computation pattern being applied
 * - Associated data transfer operations for halo updates
 */
struct Token
{
   public:
    Token() = delete;

    /**
     * @brief Construct a new Token with specified parameters
     * 
     * @param m_uid Unique identifier for the multi-GPU data being accessed
     * @param m_access Type of data access (read, write, read-write)
     * @param m_compute Computation pattern applied to the data
     */
    Token(Neon::set::dataDependency::MultiXpuDataUid m_uid,
          Neon::set::dataDependency::AccessType      m_access,
          Neon::Pattern                              m_compute);

    /**
     * @brief Update all token parameters
     * 
     * Modifies the token's data identifier, access type, and computation pattern.
     * This method also resets the data transfer container to a default error state.
     * 
     * @param m_uid New unique identifier for the multi-GPU data
     * @param m_access New type of data access
     * @param m_compute New computation pattern
     */
    auto update(Neon::set::dataDependency::MultiXpuDataUid m_uid,
                Neon::set::dataDependency::AccessType      m_access,
                Neon::Pattern                              m_compute)
        -> void;

    /**
     * @brief Get the unique identifier for the multi-GPU data
     * 
     * Returns the unique identifier that distinguishes this data object
     * across all GPUs in the system.
     * 
     * @return Unique identifier for the multi-GPU data being accessed
     */
    auto uid()
        const -> Neon::set::dataDependency::MultiXpuDataUid;

    /**
     * @brief Get the data access type
     * 
     * Returns the type of access pattern for this token, which determines
     * how the kernel interacts with the data (read-only, write-only, or read-write).
     * 
     * @return The access type specifying how the data is used
     */
    auto access()
        const -> Neon::set::dataDependency::AccessType;

    /**
     * @brief Get the computation pattern
     * 
     * Returns the pattern that describes how the computation is applied
     * to the data, such as stencil operations, map operations, etc.
     * 
     * @return The computation pattern associated with this token
     */
    auto compute()
        const -> Neon::Pattern;

    /**
     * @brief Convert the token to a human-readable string
     * 
     * Creates a string representation of the token including its data UID,
     * access type, and computation pattern. Useful for debugging and logging.
     * 
     * @return String representation of the token
     */
    auto toString()
        const -> std::string;

    /**
     * @brief Set the data transfer container for halo updates
     * 
     * Associates a container creation function that generates data transfer
     * operations for halo updates between neighboring domains in multi-GPU
     * computations.
     * 
     * @param huPerDevice Function that creates a container for data transfer
     *                    operations given a specific transfer mode
     */
    auto setDataTransferContainer(std::function<Neon::set::Container(Neon::set::TransferMode transferMode)> huPerDevice)
        -> void;

    /**
     * @brief Get the data transfer container for a specific transfer mode
     * 
     * Returns a container that performs data transfers between GPUs for the
     * specified transfer mode. This is used for halo updates in stencil
     * computations and other multi-GPU data exchange operations.
     * 
     * @param transferMode The mode of data transfer (e.g., PUT, GET)
     * @return Container that performs the requested data transfer operation
     * @throws NeonException if no data transfer container has been set
     */
    auto getDataTransferContainer(Neon::set::TransferMode transferMode)
        const -> Neon::set::Container;

    /**
     * @brief Merge another access type with the current one
     * 
     * Combines the current access type with the provided access type,
     * typically resulting in a more permissive access pattern (e.g., merging
     * READ and WRITE results in READ_WRITE).
     * 
     * @param tomerge The access type to merge with the current access type
     */
    auto mergeAccess(AccessType tomerge)
        -> void;


   private:
    Neon::set::dataDependency::MultiXpuDataUid mUid;
    Neon::set::dataDependency::AccessType      mAccess;
    Neon::Pattern                              mCompute;

    std::function<Neon::set::Container(Neon::set::TransferMode transferMode)> mHaloUpdateExtractor;
};

}  // namespace Neon::set::dataDependency
