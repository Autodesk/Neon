#pragma once

#include "Neon/set/Backend.h"
#include "Neon/set/BlockConfig.h"
#include "Neon/set/LaunchParameters.h"

namespace Neon {
namespace set {

/**
 * Abstraction to store all data needed to lunch a kernel.
 * The only data that this abstraction does not stores are
 * the kernel function and the parameters.
 */
class KernelConfig
{
   public:
    virtual ~KernelConfig() = default;

   private:
    Neon::DataView  m_dataView{Neon::DataView::STANDARD};
    Neon::Backend   m_bk;
    int             m_streamIdx = {-1};
    LaunchParameters   m_launchInfoSet;
    //--------------------------------------------------------------------------
    // INITIALIZATION
    //--------------------------------------------------------------------------
   public:
    /**
     * Empty constructor
     */
    KernelConfig() = default;

    /**
     *
     * @param bk
     * @param streamIdx
     * @param launchInfoSet
     */
    KernelConfig(const Neon::Backend& bk,
                   int                       streamIdx,
                   const LaunchParameters&    launchInfoSet);

    /**
     *
     * @param dataView
     * @param bk
     * @param streamIdx
     * @param launchInfoSet
     */
    KernelConfig(Neon::DataView            dataView,
                   const Neon::Backend& bk,
                   int                       streamIdx,
                   const LaunchParameters&    launchInfoSet);

    /**
     * Return const reference to the dataView
     * @return
     */
    auto dataView() const -> const Neon::DataView&;

    /**
     * Return const reference to the backend
     * @return
     */
    auto backend() const -> const Neon::Backend&;

    /**
     * Return const reference to the stream
     */
    auto stream() const -> const int&;

    /**
     * Return const reference to the stream
     */
    auto streamSet() const -> const StreamSet&;

    /**
     * Return const reference to the launchInfoSet
     */
    auto launchInfoSet() const -> const LaunchParameters&;

    auto runMode() const -> Neon::run_et::et;

    /**
     * Set the dataView
     * @return
     */
    auto expertSetDataView(const Neon::DataView&) -> void;

    /**
     * Set the backend
     * @return
     */
    auto expertSetBackend(const Neon::Backend& bk) -> void;

    /**
     * Set the stream
     */
    auto expertSetStream(const int&) -> void;

    /**
     * Set the launchInfoSet
     */
    auto expertSetLaunchParameters(const LaunchParameters& l) -> void;

    auto expertSetLaunchParameters(std::function<void(LaunchParameters&)> f) -> void;
};


}  // namespace set
}  // namespace Neon
