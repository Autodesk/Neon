#pragma once
#include "Neon/set/Backend.h"
#include "Neon/set/MultiXpuDataInterface.h"
#include "Neon/set/patterns/BlasSet.h"

namespace Neon {

template <typename T>
class PatternScalar
    : public set::interface::MultiXpuDataInterface<PatternScalar<T>, int>
{

   public:
    using Partition = PatternScalar<T>;

    PatternScalar() = default;

    PatternScalar(const PatternScalar& other) = default;

    /**
     * Accessing the result of the pattern
     */
    auto operator()() -> T&;

    /**
     * Accessing the result of the pattern
     */
    auto operator()() const -> const T&;


    /**
     * Constructor which initializes internal data. Should be called by the grid
     */
    PatternScalar(Neon::Backend               backend /**< backend for allocating temp memory*/,
                  Neon::sys::patterns::Engine engine = Neon::sys::patterns::Engine::cuBlas);

    /**
     * Returns a unique identifier to be used for the loading process
     */
    auto uid() const -> Neon::set::dataDependency::MultiXpuDataUid;

    auto getPartition(const Neon::DeviceType& devType,
                      const Neon::SetIdx&     idx,
                      const Neon::DataView&   dataView = Neon::DataView::STANDARD) const
        -> const Partition&;

    auto getPartition(const Neon::DeviceType& devType,
                      const Neon::SetIdx&     idx,
                      const Neon::DataView&   dataView = Neon::DataView::STANDARD)
        -> Partition&;

    auto getPartition(Neon::Place  execution,
                      Neon::SetIdx          setIdx,
                      const Neon::DataView& dataView = Neon::DataView::STANDARD) const
        -> const Partition& final;

    auto getPartition(Neon::Place  execution,
                      Neon::SetIdx          setIdx,
                      const Neon::DataView& dataView = Neon::DataView::STANDARD)
        -> Partition& final;

    /**
     * Set what stream the computation will run on
     */
    auto setStream(int                   streamIdx,
                   const Neon::DataView& dataView) const -> void;

    /**
     * Return temporary memory used internally during the computation
     */
    auto getTempMemory(const Neon::DataView& dataView,
                       Neon::DeviceType      devType = Neon::DeviceType::CPU) -> Neon::set::MemDevSet<T>&;

    /**
     * Return Blas handle to be used internally for the computation
     */
    auto getBlasSet(const Neon::DataView& dataView) -> Neon::set::patterns::BlasSet<T>&;

    /**
     * Accessing the result of the pattern based on the data view
     */
    auto operator()(const Neon::DataView& dataView) -> T&;

    auto getName() const -> std::string;

   private:
    auto updateIO(int streamId = 0)
        -> void final;

    auto updateCompute(int streamId = 0)
        -> void final;

    struct Data
    {
        // Temp memory needed for cublas/cub to do reduction on boundary, internal, and standard data view
        Neon::set::MemDevSet<T>         hostTempBoundary;
        Neon::set::MemDevSet<T>         hostTempInternal;
        Neon::set::MemDevSet<T>         hostTempStandard;
        Neon::set::MemDevSet<T>         deviceTempBoundary;
        Neon::set::MemDevSet<T>         deviceTempInternal;
        Neon::set::MemDevSet<T>         deviceTempStandard;
        Neon::set::patterns::BlasSet<T> blasSetBoundary;
        Neon::set::patterns::BlasSet<T> blasSetInternal;
        Neon::set::patterns::BlasSet<T> blasSetStandard;
        Neon::DeviceType                devType;
        Neon::Backend                   backend;
    };
    std::shared_ptr<Data> mData;

    T boundaryResult;
    T internalResult;
    T standardResult;
};



}  // namespace Neon

#include "Neon/domain/patterns/PatternScalar_imp.h"