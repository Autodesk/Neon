#pragma once

#include "GraphDependencyType.h"
#include "Neon/set/dependency/Token.h"

namespace Neon::set::container {

struct GraphDependency
{
    std::string getLabel();


   public:
    GraphDependency();
    explicit GraphDependency(GraphDependencyType type);

    explicit GraphDependency(const Neon::set::dataDependency::Token& type);

    explicit GraphDependency(const std::vector<Neon::set::dataDependency::Token>& type);

    auto setType(GraphDependencyType type)
        -> void;

    auto getType() const
        -> GraphDependencyType;

    auto addToken(const Neon::set::dataDependency::Token& token)
        -> void;

    /**
     * Returns the tokens generating this dependency
     */
    auto getTokens()
        const -> const std::vector<Neon::set::dataDependency::Token>&;

    auto hasStencilDependency()
        const -> bool;

   private:
    GraphDependencyType                           mType;
    std::vector<Neon::set::dataDependency::Token> mTokens /**< Tokens creating this dependency. */;
    // TODO - add information for data and Scheduling dependency
};

}  // namespace Neon::set::container
