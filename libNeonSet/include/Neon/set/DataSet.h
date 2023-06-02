#pragma once
#include <vector>

#include "Neon/core/types/DataView.h"
#include "Neon/set/MultiXpuDataUid.h"
#include "Neon/sys/devices/DevInterface.h"

namespace Neon {
namespace set {

template <typename T_ta>
struct DataSet
{
   private:
    std::shared_ptr<std::vector<T_ta>> m_data;

    /**
     * Returns a mutable reference to the data hold by the object
     */
    auto h_vec()
        -> std::vector<T_ta>&
    {
        return *(m_data.get());
    }

    /**
     * Returns a constant reference to the data hold by the object
     */
    auto h_vec()
        const
        -> const std::vector<T_ta>&
    {
        return *(m_data.get());
    }

   public:
    using local_t = T_ta;

    /**
     * Empty constructor
     */
    DataSet()
    {
        m_data = std::make_shared<std::vector<T_ta>>();
    }

    /**
     * Returns a reference to this object
     * @return
     */
    auto self() -> DataSet&
    {
        return *this;
    }

    /**
     * Returns a constant reference to this object
     * @return
     */
    auto self() const -> const DataSet&
    {
        return *this;
    }

    /**
     *
     * @param nDev
     */
    DataSet(int nDev)
    {
        m_data = std::make_shared<std::vector<T_ta>>(nDev);
    }

    /**
     *
     * @param nDev
     * @param val
     */
    DataSet(int nDev, T_ta val)
    {
        m_data = std::make_shared<std::vector<T_ta>>(nDev, val);
    }

    /**
     *
     * @param vec
     */
    DataSet(const std::vector<T_ta>& vec)
    {
        m_data = std::make_shared<std::vector<T_ta>>(vec);
    }

    /**
     * Deep copy method
     * @return
     */
    auto clone() const -> DataSet
    {
        DataSet cloned(*m_data);
        return cloned;
    }
    /**
     * Returns a unique identifier for the data set.
     * @return
     */
    auto uid()
        const -> Neon::set::dataDependency::MultiXpuDataUid
    {
        T_ta* addr = m_data.get();
        auto  uidRes = (Neon::set::dataDependency::MultiXpuDataUid)addr;
        return uidRes;
    }

    /**
     *
     * @return
     */
    auto getPartition(Neon::Execution, SetIdx setIdx, const Neon::DataView& dataView = Neon::DataView::STANDARD)
        -> T_ta&
    {
        (void)dataView;
        return h_vec().at(setIdx);
    }

    /**
     *
     * @return
     */
    auto getPartition(Neon::Execution, SetIdx setIdx, const Neon::DataView& dataView = Neon::DataView::STANDARD) const -> const T_ta&
    {
        (void)dataView;
        return h_vec().at(setIdx);
    }

    template <typename NewType>
    auto typedClone() const -> Neon::set::DataSet<NewType>
    {
        Neon::set::DataSet<NewType> result (static_cast<int>(m_data->size()));
        result.forEachSeq([&](Neon::SetIdx const& setIdx,
                              NewType& val ){
            val = this->operator[](setIdx);
        });
        return result;
    }

    /**
     *
     * @param idx
     * @return
     */
    auto operator[](int64_t idx) const -> const T_ta&
    {
        return h_vec().at(idx);
    }

    /**
     *
     * @param idx
     * @return
     */
    auto operator[](int64_t idx) -> T_ta&
    {
        return h_vec().at(idx);
    }

    /**
     *
     * @return
     */
    auto empty() const -> bool
    {
        return h_vec().empty();
    }

    /**
     *
     * @return
     */
    auto size() const -> int64_t
    {
        return static_cast<int64_t>(h_vec().size());
    }


    auto begin() const -> const typename std::vector<T_ta>::const_iterator
    {
        return h_vec().begin();
    }

    auto begin() -> typename std::vector<T_ta>::iterator
    {
        return h_vec().begin();
    }

    auto end() const -> const typename std::vector<T_ta>::const_iterator
    {
        return h_vec().end();
    }

    auto end() -> typename std::vector<T_ta>::iterator
    {
        return h_vec().end();
    }

    auto vec()
        const
        -> const std::vector<T_ta>&
    {
        return h_vec();
    }

    auto cardinality() -> int
    {
        return int(h_vec().size());
    }

    auto cardinality() const -> int
    {
        return int(h_vec().size());
    }

    template <typename newType_ta>
    auto newType() const -> DataSet<newType_ta>
    {
        DataSet<newType_ta> newDataSet(self().cardinality());
        for (SetIdx id = 0; id < self().cardinality(); id++) {
            newDataSet[id] = newType_ta(self()[id]);
        }
        return newDataSet;
    }

    template <typename Lambda>
    auto forEachSeq(Lambda const& fun)
    {
        for (SetIdx id = 0; id < self().cardinality(); id++) {
            fun(id, self()[id]);
        }
    }

    template <typename Lambda>
    auto forEachSeq(Lambda const& fun) const
    {
        for (SetIdx id = 0; id < self().cardinality(); id++) {
            fun(id, self()[id]);
        }
    }

    template <typename Lambda>
    auto forEachPar(Lambda const& fun)
    {
        int cardCount = self().cardinality();
#pragma omp parallel for num_threads(cardCount)
        for (int id = 0; id < self().cardinality(); id++) {
            fun(id, self()[id]);
        }
    }
};

template <typename T_ta>
auto begin(DataSet<T_ta>& data) -> typename std::vector<T_ta>::iterator
{
    return data.begin();
}

template <typename T_ta>
auto end(DataSet<T_ta>& data) -> typename std::vector<T_ta>::iterator
{
    return data.end();
}

template <typename T_ta>
auto begin(const DataSet<T_ta>& data) -> typename std::vector<T_ta>::const_iterator
{
    return data.begin();
}

template <typename T_ta>
auto end(const DataSet<T_ta>& data) -> typename std::vector<T_ta>::const_iterator
{
    return data.end();
}

template <typename... funVectorInputType_ta>
struct DataSetInput_t
{
    std::tuple<Neon::set::DataSet<funVectorInputType_ta>&...> t;

    DataSetInput_t() = delete;

    explicit DataSetInput_t(Neon::set::DataSet<funVectorInputType_ta>&... args)
        : t(args...)
    {
    }
};

template <typename... funVectorInputType_ta>
DataSetInput_t<funVectorInputType_ta...> dataSetInput(Neon::set::DataSet<funVectorInputType_ta>&... args)
{
    return DataSetInput_t<funVectorInputType_ta...>(args...);
}


}  // namespace set
}  // namespace Neon
