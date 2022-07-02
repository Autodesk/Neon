#pragma once

#include "Neon/domain/tools/PointHashTableSet.h"

namespace Neon::domain::tool {

template <typename IntegerT, typename MetaT>
PointHashTableSet<IntegerT, MetaT>::PointHashTableSet()
    : mNumDevices(0)
{
}

template <typename IntegerT, typename MetaT>
PointHashTableSet<IntegerT, MetaT>::PointHashTableSet(const Neon::domain::interface::GridBase& baseGrid)
    : mNumDevices(baseGrid.getDevSet().setCardinality()),
      mBbox(baseGrid.getDimension())
{
    mTablesSetDw = baseGrid.getDevSet().template newDataSet<std::array<HashTable, HelpNumOptions>>();
    for (int i = 0; i < mNumDevices; i++) {
        HashTable& internal = mTablesSetDw[i][HelpInternal];
        HashTable& boundary = mTablesSetDw[i][HelpBoundary];

        internal = HashTable(baseGrid.getDimension());
        boundary = HashTable(baseGrid.getDimension());
    }
}

template <typename IntegerT, typename MetaT>
auto PointHashTableSet<IntegerT, MetaT>::getMetadata(Point const& point, SetIdx& setIdx, DataView& dw) const
    -> Meta const*
{
    Meta* meta = nullptr;
    for (int i = 0; i < mNumDevices; i++) {
        HashTable& internal = mTablesSetDw[i][HelpInternal];
        HashTable& boundary = mTablesSetDw[i][HelpBoundary];

        auto* tmp = internal.getMetadata(point);
        if (meta != nullptr) {
            meta = tmp;
            dw = Neon::DataView::INTERNAL;
            setIdx = i;
            break;
        }

        tmp = boundary.getMetadata(point);
        if (meta != nullptr) {
            meta = tmp;
            dw = Neon::DataView::BOUNDARY;
            setIdx = i;
            break;
        }
    }
    return meta;
}

template <typename IntegerT, typename MetaT>
auto PointHashTableSet<IntegerT, MetaT>::getMetadata(Point const& point, SetIdx& setIdx, DataView& dw)
    -> Meta*
{
    Meta* meta = nullptr;
    for (int i = 0; i < mNumDevices; i++) {
        HashTable& internal = mTablesSetDw[i][HelpInternal];
        HashTable& boundary = mTablesSetDw[i][HelpBoundary];

        auto* tmp = internal.getMetadata(point);
        if (tmp != nullptr) {
            meta = tmp;
            dw = Neon::DataView::INTERNAL;
            setIdx = i;
            break;
        }

        tmp = boundary.getMetadata(point);
        if (tmp != nullptr) {
            meta = tmp;
            dw = Neon::DataView::BOUNDARY;
            setIdx = i;
            break;
        }
    }
    return meta;
}

template <typename IntegerT, typename MetaT>
auto PointHashTableSet<IntegerT, MetaT>::HelpFromDataViewToLocalNaming(DataView dw)
    -> int
{
    if (DataView::INTERNAL == dw) {
        return HelpInternal;
    }
    if (DataView::BOUNDARY == dw) {
        return HelpBoundary;
    }
    NEON_THROW_UNSUPPORTED_OPTION("PointHashTableSet");
}

template <typename IntegerT, typename MetaT>
auto PointHashTableSet<IntegerT, MetaT>::addPoint(const PointHashTableSet::Point& p,
                                                  const Meta&                     meta,
                                                  const SetIdx&                   setIdx,
                                                  const DataView&                 dw) -> void
{
    mTablesSetDw[setIdx][HelpFromDataViewToLocalNaming(dw)].addPoint(p, meta);
}

template <typename IntegerT, typename MetaT>
template <typename UserLambda>
auto PointHashTableSet<IntegerT, MetaT>::forEach(const UserLambda& userLambda)
{
    for (int i = 0; i < mNumDevices; i++) {
        HashTable& internal = mTablesSetDw[i][HelpInternal];
        HashTable& boundary = mTablesSetDw[i][HelpBoundary];

        internal.forEach([&](Meta& meta) {
            userLambda(i, DataView::INTERNAL, meta);
        });
        boundary.forEach([&](Meta& meta) {
            userLambda(i, DataView::BOUNDARY, meta);
        });
    }
    return;
}

template <typename IntegerT, typename MetaT>
template <typename UserLambda>
auto PointHashTableSet<IntegerT, MetaT>::forEach(Neon::SetIdx      setIdx,
                                                 DataView          dataView,
                                                 const UserLambda& userLambda)
{
    HashTable* hashTable = nullptr;
    if (dataView == DataView::INTERNAL) {
        hashTable = &mTablesSetDw[setIdx][HelpInternal];
    }
    if (dataView == DataView::BOUNDARY) {
        hashTable = &mTablesSetDw[setIdx][HelpBoundary];
    }
    if (hashTable == nullptr) {
        NEON_THROW_UNSUPPORTED_OPERATION("PointHashTableSet");
    }
    hashTable->forEach([&](const Point& point, Meta& meta) {
        userLambda(point, meta);
    });
}

}  // namespace Neon::domain::tool