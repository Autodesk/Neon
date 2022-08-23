#pragma once

#include "Neon/set/container/ContainerOperationType.h"
#include "Neon/set/container/ContainerPatternType.h"

#include "Neon/set/DevSet.h"
#include "Neon/set/container/ContainerExecutionType.h"
#include "Neon/set/dependencyTools/DataParsing.h"

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
     * Returns a name associated to the container
     */
    auto getName() const
        -> const std::string&;

    auto getTokens() const
        -> const std::vector<Neon::set::internal::dependencyTools::DataToken>&;

    auto getTokenRef()
        -> std::vector<Neon::set::internal::dependencyTools::DataToken>&;

    /**
     * Get the execution type for the container
     */
    auto getContainerExecutionType() const -> Neon::set::ContainerExecutionType;

    /**
     * Get the Operation type for the container
     */
    auto getContainerOperationType() const -> Neon::set::ContainerOperationType;

    /**
     * Get the Pattern type for the container
     */
    auto getContainerPatternType() const -> Neon::set::ContainerPatternType;

    auto getDataViewSupport() const
        -> DataViewSupport;

    /**
     * Log information on the parsed tokens.
     */
    auto toLog(uint64_t ContainerUid) -> void;

   protected:
    /**
     * Set the name for the container
     */
    auto setName(const std::string& name)
        -> void;

    /**
     * Set the launch parameters for the container
     * @param dw
     * @return
     */
    auto setLaunchParameters(Neon::DataView dw)
        -> Neon::set::LaunchParameters&;

    /**
     * Get the launch parameters for the container
     */
    auto getLaunchParameters(Neon::DataView dw) const
        -> const Neon::set::LaunchParameters&;

    /**
     * Set the execution type for the container
     */
    auto setContainerExecutionType(Neon::set::ContainerExecutionType containerType) -> void;

    /**
     * Set the Operation type for the container
     */
    auto setContainerOperationType(Neon::set::ContainerOperationType containerType) -> void;


    /**
     * Set the DataView support for the container
     */
    auto setDataViewSupport(DataViewSupport dataViewSupport) -> void;

    auto setContainerPattern(const std::vector<Neon::set::internal::dependencyTools::DataToken>& tokens) -> void;

    bool mParsingDataUpdated = false;

   private:
    std::vector<Neon::set::internal::dependencyTools::DataToken>         mParsed;
    std::string                                                          mName{"Anonymous"}; /**< Name of the Container */
    std::array<Neon::set::LaunchParameters, Neon::DataViewUtil::nConfig> mLaunchParameters;
    Neon::set::ContainerExecutionType                                    mContainerExecutionType;
    Neon::set::ContainerOperationType                                    mContainerOperationType;
    Neon::set::ContainerPatternType                                      mContainerPatternType;
    DataViewSupport                                                      mDataViewSupport = DataViewSupport::on;
};

}  // namespace Neon::set::internal
