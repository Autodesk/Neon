#pragma once

#include "Neon/set/container/ContainerOperationType.h"
#include "Neon/set/container/ContainerPatternType.h"

#include "Neon/set/DevSet.h"
#include "Neon/set/container/ContainerExecutionType.h"


#include "Neon/set/dependency/Token.h"
#include "functional"
#include "type_traits"

namespace Neon::set {
struct Loader;
}

namespace Neon::set::container {
struct Graph;
}

namespace Neon::set::internal {

/**
 * Abstract representation of a kernel container.
 * This abstraction is not exposed to the users.
 */
struct ContainerAPI
{
   public:
    friend Neon::set::Loader;

    enum struct DataViewSupport
    {
        on,
        off,
    };

    /**
     * virtual default destructor.
     */
    virtual ~ContainerAPI() = default;

    /**
     * Run this Container over a stream.
     */
    virtual auto run(int streamIdx, Neon::DataView dataView = Neon::DataView::STANDARD)
        -> void = 0;

    /**
     * Run this Container over a stream.
     */
    virtual auto run(Neon::SetIdx idx, int streamIdx, Neon::DataView dataView)
        -> void = 0;

    /**
     * Returns a pointer to the internal host container.
     */
    virtual auto getHostContainer()
        -> std::shared_ptr<ContainerAPI>;

    /**
     * Returns a pointer to the internal device container.
     */
    virtual auto getDeviceContainer()
        -> std::shared_ptr<ContainerAPI>;

    /**
     * Returns a handle to the internal graph of Containers.
     */
    virtual auto getGraph()
        -> const Neon::set::container::Graph&;

    /**
     * Parse the input and output data for the kernel.
     * @return
     */
    virtual auto parse()
        -> const std::vector<Neon::set::dataDependency::Token>& = 0;


    /**
     * Returns a name associated to the container.
     */
    auto getName() const
        -> const std::string&;

    /**
     * Returns a list of tokens as result of parsing the Container loading lambda.
     */
    auto getTokens() const
        -> const std::vector<Neon::set::dataDependency::Token>&;

    /**
     * Returns a list of tokens as result of parsing the Container loading lambda.
     */
    auto getTokenRef()
        -> std::vector<Neon::set::dataDependency::Token>&;

    /**
     * Get the execution type for the Container.
     */
    auto getContainerExecutionType() const
        -> Neon::set::ContainerExecutionType;

    /**
     * Get the Operation type for the Container.
     */
    auto getContainerOperationType() const
        -> Neon::set::ContainerOperationType;

    /**
     * Get the Pattern type for the Container
     */
    auto getContainerPatternType() const
        -> Neon::set::ContainerPatternType;

    /**
     * Returns information about DataView support for this Container.
     */
    auto getDataViewSupport() const
        -> DataViewSupport;

    /**
     * Log information on the parsed tokens.
     */
    auto toLog(uint64_t ContainerUid) -> void;

   protected:
    /**
     * Add a new token
     */
    auto addToken(Neon::set::dataDependency::Token& dataParsing)
        -> void;

    /**
     * Set the name for the container
     */
    auto setName(const std::string& name)
        -> void;

    /**
     * Set the launch parameters for the container
     */
    auto setLaunchParameters(Neon::DataView dw)
        -> Neon::set::LaunchParameters&;

    /**
     * Get the launch parameters for the container
     */
    auto getLaunchParameters(Neon::DataView dw) const
        -> const Neon::set::LaunchParameters&;

    /**
     * Generate a string that will be printed in case or exceptions
     * @return
     */
    auto helpGetNameForError()
        -> std::string;

    /**
     * Set the execution type for the container
     */
    auto setContainerExecutionType(Neon::set::ContainerExecutionType containerType)
        -> void;

    /**
     * Set the Operation type for the container
     */
    auto setContainerOperationType(Neon::set::ContainerOperationType containerType)
        -> void;

    /**
     * Set the DataView support for the container
     */
    auto setDataViewSupport(DataViewSupport dataViewSupport)
        -> void;

    /**
     * Set the patter for this Container based on a list of tokens.
     * @param tokens
     */
    auto setContainerPattern(const std::vector<Neon::set::dataDependency::Token>& tokens)
        -> void;

    /**
     * Set the patter for this Container
     * @param tokens
     */
    auto setContainerPattern(ContainerPatternType patternType)
        -> void;

    auto isParsingDataUpdated()
        -> bool;

    auto setParsingDataUpdated(bool)
        -> void;

   private:
    using TokenList = std::vector<Neon::set::dataDependency::Token>;

    std::string                                                          mName{"Anonymous"}; /**< Name of the Container */
    bool                                                                 mParsingDataUpdated = false;
    TokenList                                                            mParsed;
    std::array<Neon::set::LaunchParameters, Neon::DataViewUtil::nConfig> mLaunchParameters;
    Neon::set::ContainerExecutionType                                    mContainerExecutionType;
    Neon::set::ContainerOperationType                                    mContainerOperationType;
    Neon::set::ContainerPatternType                                      mContainerPatternType;
    DataViewSupport                                                      mDataViewSupport = DataViewSupport::on;
};

}  // namespace Neon::set::internal
