#pragma once

#include "Neon/set/DevSet.h"
#include "Neon/set/dependencyTools/DataParsing.h"
#include "Neon/set/ContainerTools/ContainerType.h"

#include "functional"
#include "type_traits"


namespace Neon::set::internal {

/**
 * Abstract representation of a kernel container.
 * This abstraction is not exposed to the users.
 */
struct ContainerAPI
{
   public:
    enum struct DataViewSupport
    {
        on,
        off
    };

    /**
     * virtual default destructor
     */
    virtual ~ContainerAPI() = default;

    /**
     * Run the container over streams
     * @param streamIdx
     * @param dataView
     */
    virtual auto run(int streamIdx, Neon::DataView dataView = Neon::DataView::STANDARD) -> void = 0;

    virtual auto run(Neon::SetIdx idx, int streamIdx, Neon::DataView dataView) -> void = 0;

    virtual auto getHostContainer() -> std::shared_ptr<ContainerAPI> = 0;

    virtual auto getDeviceContainer() -> std::shared_ptr<ContainerAPI> = 0;

    /**
     * Parse the input and output data for the kernel
     * @return
     */
    virtual auto parse() -> const std::vector<Neon::set::internal::dependencyTools::DataToken>& = 0;

    /**
     *
     * @param dataParsing
     */
    auto addToken(Neon::set::internal::dependencyTools::DataToken& dataParsing)
        -> void;

    /**
     *
     */
    auto getName() const
        -> const std::string&;

    auto getContainerType() const
        -> ContainerType;

    auto getTokens() const
        -> const std::vector<Neon::set::internal::dependencyTools::DataToken>&;

    auto getTokenRef()
        -> std::vector<Neon::set::internal::dependencyTools::DataToken>&;

    auto getDataViewSupport() -> DataViewSupport;

    /**
     * Log information on the parsed tokens.
     */
    auto toLog(uint64_t ContainerUid)->void;

   protected:
    auto setName(const std::string& name)
        -> void;

    auto setLaunchParameters(Neon::DataView dw)
        -> Neon::set::LaunchParameters&;

    auto getLaunchParameters(Neon::DataView dw) const
        -> const Neon::set::LaunchParameters&;

    auto setContainerType(ContainerType containerType) -> void;

    auto setDataViewSupport(DataViewSupport dataViewSupport) -> void;


   private:
    std::vector<Neon::set::internal::dependencyTools::DataToken>         mParsed;
    std::string                                                          mName{"Anonymous"}; /**< Name of the Container */
    std::array<Neon::set::LaunchParameters, Neon::DataViewUtil::nConfig> mLaunchParameters;
    ContainerType                                                        mContainerType;
    DataViewSupport                                                      mDataViewSupport = DataViewSupport::on;
};

}  // namespace Neon::set::internal