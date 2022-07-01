#pragma once
#include <cublas_v2.h>

#include "Neon/sys/devices/gpu/GpuStream.h"
#include "Neon/sys/memory/MemDevice.h"

namespace Neon::sys::patterns {

enum class Engine
{
    cuBlas = 0,
    CUB = 1,
};

/**
 * Collection of computational patterns computed using cuBLAS. This class is not thread safe as it assumes only a single host thread to call it
*/
template <typename T /**< The input/output data type. Only float and double are allowed */>
class Blas
{

   public:
    Blas() = default;


    /**
     * Constructor which initializes internal data i.e., cuBLAS handle on the given device
    */
    Blas(const Neon::sys::GpuDevice& dev /**< The device on which Blas will run*/,
         const Neon::DeviceType&     devType = Neon::DeviceType::CUDA, /**< The device type*/
         const Engine                engine = Engine::cuBlas /**< the type of the backend engine */);

    /**
     * Set the working stream for the pattern. It needs to be set only once unless the stream needs to be changes      
    */
    void setStream(Neon::sys::GpuStream& stream);

    /**
     * Return the stream used for computations 
    */
    const Neon::sys::GpuStream& getStream() const;

    /**
     * For CUB engine, we use number of blocks for the second phase of reduction operations.     
    */
    void setNumBlocks(const uint32_t numBlocks);

    /**
     * Return the number of blocks used by CUB engine      
    */
    uint32_t getNumBlocks() const;

    /**
     * Compute the absolute sum of the input buffer i.e. sum_{i=0}^{n-1}(abs(input[i]))
     * where n is the input size
    */
    void absoluteSum(const MemDevice<T>& input /**< input buffer. Should be allocated on the device */,
                     MemDevice<T>&       output /**< output buffer. Its size should be >=1 and it should be allocated on the device */,
                     int                 start_id = 0 /**< index of the first element where computation should be done*/,
                     int                 num_elements = std::numeric_limits<int>::max() /**< number of elements where computation should be done starting from start_id*/);

    /**
     * Compute the dot product of the input buffers i.e. sum_{i=0}^{n-1}(input1[i]*input2[i])
     * where n is the input size. Inputs should have matching size.
    */
    void dot(const MemDevice<T>& input1 /**< first input buffer. Should be allocated on the device */,
             const MemDevice<T>& input2 /**< second input buffer. Should be allocated on the device */,
             MemDevice<T>&       output /**< output buffer. Its size should be >=1 and it should be allocated on the device */,
             int                 start_id /**< index of the first element where computation should be done*/,
             int                 num_elements = std::numeric_limits<int>::max() /**< number of elements where computation should be done starting from start_id*/);

    /**
     * Execute the second phase on reduction using CUB engine      
    */
    template <typename ReductionOp>
    void reducePhase2(MemDevice<T>& output, ReductionOp reduction_op, T init);

    /**
     * Get the memory buffer (allocated by Blas) that is used as output for phase 1 
     * (and input for phase 2)
    */
    MemDevice<T>& getReducePhase1Output();

    /**
     * Compute the norm2 of the input buffer i.e. sum_{i=0}^{n-1}(sqrt(input[i]*input[i]))
     * where n is the input size. 
    */
    void norm2(const MemDevice<T>& input /**< input buffer. Should be allocated on the device */,
               MemDevice<T>&       output /**< output buffer. Its size should be >=1 and it should be allocated on the device */,
               int                 start_id = 0 /**< index of the first element where computation should be done*/,
               int                 num_elements = std::numeric_limits<int>::max() /**< number of elements where computation should be done starting from start_id*/);

    /**
     * Destructor which releases all internal data i.e., destroy cuBLAS handle 
    */
    virtual ~Blas() noexcept(false);

   private:
    /**
     * Throw an exception input/output allocators are invalid
    */
    void checkAllocator(const MemDevice<T>& input, const MemDevice<T>& output);

    std::shared_ptr<cublasHandle_t> mHandle;                /**< cuBLAS handle */
    Neon::DeviceType                mDevType;               /** < type of the device*/
    DeviceID                        mDevID;                 /** < the device on which the computation and temp memory allocations happens*/
    Neon::sys::GpuStream            mStream;                /** < stream used for the computations */
    Engine                          mEngine;                /** < the backend engine used for reduction*/
    uint32_t                        mNumBlocks;             /** < Number of blocks used to allocate and run phase 2 of reduction operations*/
    MemDevice<T>                    mDevice1stPhaseOutput;  /** < To store the results of the first phase which is the input of second phase*/
    size_t                          mDeviceCUBTempMemBytes; /** < size of the temp memory used by CUB during the second phase */
    MemDevice<T>                    mDeviceCUBTempMem;      /** <  temp memory used by CUB during the second phase */
};

}  // namespace Neon::sys::patterns

#include "Neon/sys/patterns/Blas_imp.h"