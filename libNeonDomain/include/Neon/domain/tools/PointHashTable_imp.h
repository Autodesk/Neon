#pragma once

#include "Neon/domain/tools/PointHashTable.h"

namespace Neon::domain::tool {


template <typename IntegerT, typename MetaT>
PointHashTable<IntegerT, MetaT>::PointHashTable()
    : mBBox(Point(Integer(0), Integer(0), Integer(0)))
{
}

template <typename IntegerT, typename MetaT>
PointHashTable<IntegerT, MetaT>::PointHashTable(Point const& bbox)
    : mBBox(bbox)
{
}

template <typename IntegerT, typename MetaT>
auto PointHashTable<IntegerT, MetaT>::getMetadata(Point const& point) const
    -> Meta const*
{
    const size_t key = point.mPitch(mBBox);
    auto         it = mMap.find(key);
    if (it == mMap.end()) {
        return nullptr;
    }
    return &it->second;
}

template <typename IntegerT, typename MetaT>
auto PointHashTable<IntegerT, MetaT>::getMetadata(Point const& point)
    -> Meta*
{
    const Key key = helpGetKey(point);
    auto      it = mMap.find(key);
    if (it == mMap.end()) {
        return nullptr;
    }
    return &it->second;
}

template <typename IntegerT, typename MetaT>
auto PointHashTable<IntegerT, MetaT>::addPoint(const Point& point,
                                               const Meta&  data)
    -> void
{
    const Key key = helpGetKey(point);
    mMap.insert({key, data});
}

template <typename IntegerT, typename MetaT>
auto PointHashTable<IntegerT, MetaT>::helpGetKey(const Point& point) -> Key
{
    const Key key = point.mPitch(mBBox);
    return key;
}

template <typename IntegerT, typename MetaT>
auto PointHashTable<IntegerT, MetaT>::helpGetPoint(const Key& key) -> Point
{
    Integer    d1Key = Integer(key);
    const auto d3Point = mBBox.mapTo3dIdx(d1Key);
    return d3Point;
}

template <typename IntegerT, typename MetaT>
template <typename UserLambda>
auto PointHashTable<IntegerT, MetaT>::forEach(const UserLambda& f)
{
    auto it = mMap.begin();
    while (it != mMap.end()) {

        const Key&  key = it->first;
        Meta&       meta = it->second;
        const Point point = helpGetPoint(key);
        f(point, meta);

        it++;
    }
}
}  // namespace Neon::domain::tool