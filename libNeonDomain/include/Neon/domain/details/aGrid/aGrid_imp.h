#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"

#include "Neon/sys/memory/MemDevice.h"

#include "Neon/set/BlockConfig.h"
#include "Neon/set/DevSet.h"

#include "Neon/domain/interface/GridBase.h"
#include "Neon/domain/interface/KernelConfig.h"

#include "Neon/domain/details/aGrid/aField.h"
#include "Neon/domain/details/aGrid/aPartition.h"

namespace Neon::domain::details::aGrid {

template <typename T, int C>
auto aGrid::newField(const std::string   fieldUserName,
                     int                 cardinality,
                     T                   inactiveValue,
                     Neon::DataUse       dataUse,
                     Neon::MemoryOptions memoryOptions) const
    -> aGrid::Field<T, C>
{
    memoryOptions = getDevSet().sanitizeMemoryOption(memoryOptions);
    constexpr Neon::domain::haloStatus_et::e haloStatus = Neon::domain::haloStatus_et::e::ON;

    if (C != 0 && cardinality != C) {
        NeonException exception("Dynamic and static cardinality values do not match.");
        NEON_THROW(exception);
    }
    aField<T, C> field(fieldUserName, *this, cardinality, inactiveValue,
                       haloStatus, dataUse, memoryOptions);
    return field;
}

template <typename LoadingLambda>
auto aGrid::getContainer(const std::string& name,
                         LoadingLambda      lambda)
    const
    -> Neon::set::Container
{
    Neon::domain::KernelConfig kernelConfig(0);

    const Neon::index_3d& defaultBlockSize = getDefaultBlock();
    Neon::set::Container  kContainer = Neon::set::Container::factory(name,
                                                                     Neon::set::internal::ContainerAPI::DataViewSupport::on,
                                                                     *this,
                                                                     lambda,
                                                                     defaultBlockSize,
                                                                     [](const Neon::index_3d&) { return size_t(0); });
    return kContainer;
}

template <typename LoadingLambda>
auto aGrid::getContainer(const std::string& name,
                         index_3d           blockSize,
                         size_t             sharedMem,
                         LoadingLambda      lambda)
    const
    -> Neon::set::Container
{
    Neon::domain::KernelConfig kernelConfig(0);

    const Neon::index_3d& defaultBlockSize = getDefaultBlock();
    Neon::set::Container  kContainer = Neon::set::Container::factory(name,
                                                                     *this,
                                                                     lambda,
                                                                     blockSize,
                                                                     [sharedMem](const Neon::index_3d&) { return sharedMem; });
    return kContainer;
}


}  // namespace Neon::domain::details::aGrid
