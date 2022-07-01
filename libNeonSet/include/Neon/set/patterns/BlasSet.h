#pragma once
#include "Neon/sys/patterns/Blas.h"

#include "Neon/set/GpuStreamSet.h"
#include "Neon/set/memory/memDevSet.h"
#include "Neon/sys/patterns/Blas.h"

namespace Neon::set::patterns {
/**
 * Collection of computational patterns computed using Blas from the libNeonSys. 
 This class is not thread safe as it assumes only a single host thread to call it while it allow parallelization via steams (using StreamSet)
*/
template <typename T>
class BlasSet
{

   public:
    BlasSet() = default;
    BlasSet(const BlasSet& other) = default;

    /**
     * Constructor which initializes internal data i.e., allocation for mBlasVec 
    */
    BlasSet(const Neon::set::DevSet&          devSet, /**< the set devices that will execute the pattern*/
            const Neon::sys::patterns::Engine engine = Neon::sys::patterns::Engine::cuBlas /**< the type of the backend engine */);

    /**
     * Set the working stream for each device. It needs to be set only once unless the stream needs to be changes      
    */
    void setStream(Neon::set::StreamSet& streams);

    /**
     * Retrieve the streams used by this instance  
    */
    Neon::set::StreamSet& getStream();


    /**     
     * Return the i-th Blas 
    */
    Neon::sys::patterns::template Blas<T>& getBlas(size_t i);

    /**
     * Compute the absolute sum of the input buffer i.e. sum_{i=0}^{n-1}(abs(input[i]))
     * where n is the input size over all devices.
     * @return the final result (by value) on the host 
     * 
    */
    T absoluteSum(const Neon::set::MemDevSet<T>& input /**< input buffers for each device. Should be allocated on the device */,
                  Neon::set::MemDevSet<T>&       output /**< output buffer for each device. Its size should be >=1 for each device and it should be allocated on the device */,
                  Neon::set::DataSet<int>&       start_id /**< starting id in input where computation should be done for each MemDev_t in input*/,
                  Neon::set::DataSet<int>&       num_elements /**< number of elements in each MemDev_t in input where computation should be done starting from the corresponding start_id*/);

    /**
     * Compute the dot product of the input buffers i.e. sum_{i=0}^{n-1}(input1[i]*input2[i])
     * where n is the input size over all devices. Inputs should have matching size.
     * @return the final result (by value) on the host 
    */
    T dot(const Neon::set::MemDevSet<T>& input1 /**< first input buffers for each device. Should be allocated on the device */,
          const Neon::set::MemDevSet<T>& input2 /**< second input buffers for each device. Should be allocated on the device */,
          Neon::set::MemDevSet<T>&       output /**< output buffer for each device. Its size should be >=1 for each device and it should be allocated on the device */,
          Neon::set::DataSet<int>&       start_id /**< starting id in input where computation should be done for each MemDev_t in input1 and input2*/,
          Neon::set::DataSet<int>&       num_elements /**< number of elements in each MemDev_t in input1 and input2 where computation should be done starting from the corresponding start_id*/);

    /**
     * Compute the norm2 of the input buffer i.e. sum_{i=0}^{n-1}(sqrt(input[i]*input[i]))
     * where n is the input size over all devices. 
     * @return the final result (by value) on the host 
    */
    T norm2(const Neon::set::MemDevSet<T>& input /**< input buffers for each device. Should be allocated on the device */,
            Neon::set::MemDevSet<T>&       output /**< output buffer for each device. Its size should be >=1 for each device and it should be allocated on the device */,
            Neon::set::DataSet<int>&       start_id /**< starting id in input where computation should be done for each MemDev_t in input*/,
            Neon::set::DataSet<int>&       num_elements /**< number of elements in each MemDev_t in input where computation should be done starting from the corresponding start_id*/);

    virtual ~BlasSet() = default;

   private:
    /**
     * Apply final aggregation on the host by collecting intermediate results on each device
    */
    template <typename IntermediateTransform, typename Aggregate, typename FinalTransform>
    void aggregator(Neon::set::MemDevSet<T>& output,
                    IntermediateTransform    intermedFunc,
                    Aggregate                aggFunc,
                    FinalTransform           finalFunc,
                    T                        neutralValue);


    std::shared_ptr<std::vector<Neon::sys::patterns::template Blas<T>>> mBlasVec;
    T                                                                   mAggregate;
    Neon::set::StreamSet                                                mStreams;
};

}  // namespace Neon::set::patterns

#include "Neon/set/patterns/BlasSet_imp.h"