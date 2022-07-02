#pragma once
#include "Neon/set/DevSet.h"
#include "Neon/set/patterns/BlasSet.h"

namespace Neon::set::patterns {

template <typename T>
BlasSet<T>::BlasSet(const Neon::set::DevSet&          devSet,
                    const Neon::sys::patterns::Engine engine)
{
    mBlasVec = std::make_shared<std::vector<Neon::sys::patterns::template Blas<T>>>(devSet.setCardinality());

    devSet.forEachSetIdxSeq([&](Neon::SetIdx& setIdx) {
        const Neon::sys::ComputeID  gpuId = devSet.devId(setIdx.idx());
        const Neon::sys::GpuDevice& dev = Neon::sys::globalSpace::gpuSysObj().dev(gpuId);
        (*mBlasVec)[setIdx.idx()] = Neon::sys::patterns::template Blas<T>(dev, devSet.type(), engine);
    });
}

template <typename T>
void BlasSet<T>::setStream(Neon::set::StreamSet& streams)
{
    mStreams = streams;
    for (int i = 0; i < static_cast<int>((*mBlasVec).size()); ++i) {
        (*mBlasVec)[i].setStream(streams.get(i));
    }
}
template <typename T>
Neon::set::StreamSet& BlasSet<T>::getStream()
{
    return mStreams;
}

template <typename T>
Neon::sys::patterns::template Blas<T>& BlasSet<T>::getBlas(size_t i)
{
    return (*mBlasVec)[i];
}

template <typename T>
T BlasSet<T>::absoluteSum(const Neon::set::MemDevSet<T>& input,
                          Neon::set::MemDevSet<T>&       output,
                          Neon::set::DataSet<int>&       start_id,
                          Neon::set::DataSet<int>&       num_elements)
{
    const int32_t numSet = static_cast<int>((*mBlasVec).size());
#pragma omp parallel for num_threads(numSet)
    for (int i = 0; i < numSet; ++i) {
        (*mBlasVec)[i].absoluteSum(input.getMemDev(i), output.getMemDev(i), start_id[i], num_elements[i]);
    }


    aggregator(
        output,
        [](T in) { return in; },
        [](T in1, T in2) { return in1 + in2; },
        [](T in) { return in; },
        T(0));

    return mAggregate;
}

template <typename T>
T BlasSet<T>::dot(const Neon::set::MemDevSet<T>& input1,
                  const Neon::set::MemDevSet<T>& input2,
                  Neon::set::MemDevSet<T>&       output,
                  Neon::set::DataSet<int>&       start_id,
                  Neon::set::DataSet<int>&       num_elements)
{
    const int32_t numSet = static_cast<int>((*mBlasVec).size());
#pragma omp parallel for num_threads(numSet)
    for (int i = 0; i < numSet; ++i) {
        (*mBlasVec)[i].dot(input1.getMemDev(i), input2.getMemDev(i), output.getMemDev(i), start_id[i], num_elements[i]);
    }

    aggregator(
        output,
        [](T in) { return in; },
        [](T in1, T in2) { return in1 + in2; },
        [](T in) { return in; },
        T(0));

    return mAggregate;
}

template <typename T>
T BlasSet<T>::norm2(const Neon::set::MemDevSet<T>& input,
                    Neon::set::MemDevSet<T>&       output,
                    Neon::set::DataSet<int>&       start_id,
                    Neon::set::DataSet<int>&       num_elements)
{
    const int32_t numSet = static_cast<int>((*mBlasVec).size());
#pragma omp parallel for num_threads(numSet)
    for (int i = 0; i < numSet; ++i) {
        (*mBlasVec)[i].norm2(input.getMemDev(i), output.getMemDev(i), start_id[i], num_elements[i]);
    }


    aggregator(
        output,
        [](T in) { return in * in; },
        [](T in1, T in2) { return in1 + in2; },
        [](T in) { return static_cast<T>(std::sqrt(in)); },
        T(0));

    return mAggregate;
}

template <typename T>
template <typename IntermediateTransform, typename Aggregate, typename FinalTransform>
void BlasSet<T>::aggregator(Neon::set::MemDevSet<T>& output,
                            IntermediateTransform    intermedFunc,
                            Aggregate                aggFunc,
                            FinalTransform           finalFunc,
                            T                        neutralValue)
{
    const int32_t numSet = static_cast<int>((*mBlasVec).size());

    if (output.allocType() == Neon::Allocator::CUDA_MEM_DEVICE ||
        output.allocType() == Neon::Allocator::CUDA_MEM_UNIFIED) {
        NeonException exc;
        exc << "Unsupported allocator for the output with type "
            << Neon::AllocatorUtils::toString(output.getMemDev(0).allocType());
        NEON_THROW(exc);
    }

    if (numSet == 1) {
        mAggregate = output.mem(0)[0];
        return;
    }
    T temp = neutralValue;
    for (int i = 0; i < numSet; ++i) {
        T val = output.mem(i)[0];
        val = intermedFunc(val);
        temp = aggFunc(val, temp);
    }

    mAggregate = finalFunc(temp);
}
}  // namespace Neon::set::patterns