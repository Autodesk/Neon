#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"

#include "Neon/set/DataConfig.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/HuOptions.h"
#include "Neon/set/MemoryTransfer.h"


#include "Neon/domain/aGrid.h"
#include "Neon/domain/interface/FieldBaseTemplate.h"
#include "Neon/domain/tools/PartitionTable.h"
#include "Neon/domain/tools/partitioning/Cassifications.h"

namespace Neon::domain::tool {

struct HaloTable1DPartitioning
{
    using TableEntry = Neon::set::DataSet<std::vector<Neon::set::MemoryTransfer>>;
    using Table =
        std::array<
            std::array<
                std::array<
                    TableEntry,
                    Neon::domain::tool::partitioning::ByDirectionUtils::nConfigs>,
                ExecutionUtils::numConfigurations>,
            Neon::set::TransferModeUtils::nOptions>;

    Table table;

    auto get(Neon::set::TransferMode   transferMode,
             Execution                 execution,
             partitioning::ByDirection byDirection) -> const TableEntry&
    {
        auto const& tableEntryRW = table[static_cast<unsigned long>(transferMode)]
                                        [static_cast<unsigned long>(execution)]
                                        [static_cast<unsigned long>(byDirection)];
        return tableEntryRW;
    }

//    auto toString() const -> std::string
//    {
//        std::stringstream msg;
//        std::string       tab = "   ";
//        using namespace Neon::domain::tool;
//        for (auto execution : {Execution::device, Execution::host}) {
//            for (auto mode : {Neon::set::TransferMode::put, Neon::set::TransferMode::get}) {
//                msg << ExecutionUtils::toString(execution) << " - PUT \n";
//                for (auto byDirection : {partitioning::ByDirection::up, partitioning::ByDirection::down}) {
//                    //                                       upOut[setIdx][static_cast<unsigned long>(transfer)][static_cast<unsigned long>(execution)],
//                    msg << tab << partitioning::ByDirectionUtils::toString(byDirection) << "\n";
//                    auto& tableEntryRW = table[static_cast<unsigned long>(Neon::set::TransferMode::put)]
//                                              [static_cast<unsigned long>(execution)]
//                                              [static_cast<unsigned long>(byDirection)];
//
//                    tableEntryRW.forEachSeq([&](SetIdx setIdx, auto& stdVecOfTRansfers) {
//                        lambda(setIdx, execution, byDirection, stdVecOfTRansfers);
//                    });
//                }
//            }
//        }
//
//        auto getNghSetIdx = [&](SetIdx setIdx, Neon::domain::tool::partitioning::ByDirection direction) {
//            int res;
//            if (direction == Neon::domain::tool::partitioning::ByDirection::up) {
//                res = (setIdx + 1) % bk.getDeviceCount();
//            } else {
//                res = (setIdx + bk.getDeviceCount() - 1) % bk.getDeviceCount();
//            }
//            return res;
//        };
//
//        for (auto execution : {Execution::device, Execution::host}) {
//            for (auto byDirection : {partitioning::ByDirection::up, partitioning::ByDirection::down}) {
//                //                                       upOut[setIdx][static_cast<unsigned long>(transfer)][static_cast<unsigned long>(execution)],
//                auto const& tableEntryRO = table[static_cast<unsigned long>(Neon::set::TransferMode::put)]
//                                                [static_cast<unsigned long>(execution)]
//                                                [static_cast<unsigned long>(byDirection)];
//                auto& tableEntryRW = table[static_cast<unsigned long>(Neon::set::TransferMode::get)]
//                                          [static_cast<unsigned long>(execution)]
//                                          [static_cast<unsigned long>(byDirection)];
//                tableEntryRW = bk.newDataSet<std::vector<Neon::set::MemoryTransfer>>();
//
//                tableEntryRW.forEachSeq([&](SetIdx setIdx, auto& stdVecOfTRansfers) {
//                    Neon::SetIdx otherSetIdx = getNghSetIdx(setIdx, byDirection);
//                    for (auto putTransfer : tableEntryRO[otherSetIdx]) {
//                        if (putTransfer.dst.setIdx == setIdx) {
//                            stdVecOfTRansfers.push_back(putTransfer);
//                        }
//                    }
//                });
//            }
//        }
//    }

    template <typename Lambda>
    auto forEachPutConfiguration(const Neon::Backend& bk, Lambda const& lambda) -> void
    {
        using namespace Neon::domain::tool;
        for (auto execution : {Execution::device, Execution::host}) {
            for (auto byDirection : {partitioning::ByDirection::up, partitioning::ByDirection::down}) {
                //                                       upOut[setIdx][static_cast<unsigned long>(transfer)][static_cast<unsigned long>(execution)],
                auto& tableEntryRW = table[static_cast<unsigned long>(Neon::set::TransferMode::put)]
                                          [static_cast<unsigned long>(execution)]
                                          [static_cast<unsigned long>(byDirection)];
                tableEntryRW = bk.newDataSet<std::vector<Neon::set::MemoryTransfer>>();

                tableEntryRW.forEachSeq([&](SetIdx setIdx, auto& stdVecOfTRansfers) {
                    lambda(setIdx, execution, byDirection, stdVecOfTRansfers);
                });
            }
        }

        auto getNghSetIdx = [&](SetIdx setIdx, Neon::domain::tool::partitioning::ByDirection direction) {
            int res;
            if (direction == Neon::domain::tool::partitioning::ByDirection::up) {
                res = (setIdx + 1) % bk.getDeviceCount();
            } else {
                res = (setIdx + bk.getDeviceCount() - 1) % bk.getDeviceCount();
            }
            return res;
        };

        for (auto execution : {Execution::device, Execution::host}) {
            for (auto byDirection : {partitioning::ByDirection::up, partitioning::ByDirection::down}) {
                //                                       upOut[setIdx][static_cast<unsigned long>(transfer)][static_cast<unsigned long>(execution)],
                auto const& tableEntryRO = table[static_cast<unsigned long>(Neon::set::TransferMode::put)]
                                                [static_cast<unsigned long>(execution)]
                                                [static_cast<unsigned long>(byDirection)];
                auto& tableEntryRW = table[static_cast<unsigned long>(Neon::set::TransferMode::get)]
                                          [static_cast<unsigned long>(execution)]
                                          [static_cast<unsigned long>(byDirection)];
                tableEntryRW = bk.newDataSet<std::vector<Neon::set::MemoryTransfer>>();

                tableEntryRW.forEachSeq([&](SetIdx setIdx, auto& stdVecOfTRansfers) {
                    Neon::SetIdx otherSetIdx = getNghSetIdx(setIdx, byDirection);
                    for (auto putTransfer : tableEntryRO[otherSetIdx]) {
                        if (putTransfer.dst.setIdx == setIdx) {
                            stdVecOfTRansfers.push_back(putTransfer);
                        }
                    }
                });
            }
        }
    }
};
}  // namespace Neon::domain::tool