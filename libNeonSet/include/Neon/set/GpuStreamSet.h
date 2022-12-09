#pragma once

#include "Neon/set/GpuEventSet.h"
#include "Neon/sys/devices/gpu/GpuStream.h"
#include <omp.h>

namespace Neon {
namespace set {

/**
 *
 */
class StreamSet
{
   public:
    // FRIENDS
    friend class DevSet;

   private:
    // MEMBERS
    std::vector<Neon::sys::GpuStream> m_streamVec;

   public:
    //--------------------------------------------------------------------------
    // Constructors
    //--------------------------------------------------------------------------
    int cardinality() const
    {
        return static_cast<int>(m_streamVec.size());
    }
    /**
     * Empty constructor
     */
    StreamSet() = default;

   private:
    /**
     * Constructor that accept the number of streams that will be stored in the object.
     * @param setCardinality
     */
    StreamSet(int setCardinality);
    /**
     * Provide read only access to a targeted GpuStream object in the set
     * @tparam accessType
     * @param id
     * @return
     */
    auto set(SetIdx id, Neon::sys::GpuStream&& gpuStream) -> void;

   public:
    /**
     * Default destructor. The object implements a automatic garbage collection,
     * therefore it there is no GpuStream pointing to the allocated CUDA stream resource
     * a release operation is executed.
     */
    virtual ~StreamSet() = default;

    //--------------------------------------------------------------------------
    // INSPECTION
    //--------------------------------------------------------------------------
    /**
     * Provide read only access to a targeted GpuStream object in the set
     * @tparam accessType
     * @param id
     * @return
     */
    template <Neon::Access accessType = Neon::Access::read>
    [[deprecated]] inline typename std::enable_if<metaProgramming::isReadOnly_type_t<accessType>::value, const Neon::sys::GpuStream&>::type stream(SetIdx id) const
    {
        if (id.idx() > int32_t(m_streamVec.size()) || m_streamVec.empty()) {
            if (m_streamVec.empty()) {
                Neon::NeonException exp("GpuStreamSet");
                exp << "Incompatible stream id " << id << ". This stream set was not initialized";
                NEON_THROW(exp);
            } else {
                Neon::NeonException exp("GpuStreamSet");
                exp << "Incompatible stream id " << id << ". Id range is: 0," << m_streamVec.size();
                NEON_THROW(exp);
            }
        }
        return m_streamVec[id.idx()];
    }

    /**
     *
     * @param id
     * @return
     */
    inline auto operator[](SetIdx id) const
        -> const Neon::sys::GpuStream&
    {
        validateId(id);
        return m_streamVec[id.idx()];
    }


    /**
     *
     * @param id
     * @return
     */
    auto get(SetIdx id) const
        -> const Neon::sys::GpuStream&;

    /**
     *
     * @param id
     * @return
     */
    auto get(SetIdx id)
        -> Neon::sys::GpuStream&;


    /**
     * Returns a reference to the cudaStream object associated with the i-th GpuStream in the set.
     * @param idx
     * @return
     */
    const cudaStream_t& cudaStream(int idx) const
    {
        return m_streamVec[idx].stream();
    }


    //--------------------------------------------------------------------------
    // RESOURCE MANAGEMENT
    //--------------------------------------------------------------------------

    /**
     * Release all the cudaStream elements that are held by the object.
     */
    void release();


    //--------------------------------------------------------------------------
    // SYNCRONIZATIONS
    //--------------------------------------------------------------------------

    template <run_et::et runMode = run_et::et::sync>
    void sync() const
    {
        const int nDev = (int)m_streamVec.size();

        // Without nDev>0, VS on debug mode displays this annoying message
        //"User Error 1001: argument to num_threads clause must be positive"
        if (run_et::et::sync == runMode && nDev > 0) {
#pragma omp parallel num_threads(nDev)
            {
                const int idx = omp_get_thread_num();
                m_streamVec[idx].sync<runMode>();
            }
            return;
        } else {
            return;
        }
    }

    template <run_et::et runMode = run_et::et::sync>
    auto sync(int idx) const -> void
    {
        m_streamVec[idx].sync<runMode>();
    }

    /**
     *
     * @param eventSet
     */
    auto enqueueEvent(GpuEventSet& eventSet) const -> void;

    /**
     *
     * @param eventSet
     */
    auto waitForEvent(GpuEventSet& eventSet) const -> void;

    /**
     *
     * @param eventSet
     */
    auto enqueueEvent(Neon::SetIdx setIdx,
                      GpuEventSet& eventSet) const -> void;

    /**
     *
     * @param eventSet
     */
    auto waitForEvent(Neon::SetIdx setIdx,
                      GpuEventSet& eventSet) const -> void;

   private:
    /**
     *
     * @param id
     */
    auto validateId(SetIdx id) const
        -> void;
};  // namespace set

}  // namespace set
}  // End of namespace Neon
