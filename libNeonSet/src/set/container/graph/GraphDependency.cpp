#include "Neon/set/container/graph/GraphDependency.h"
#include <string>

namespace Neon::set::container {

GraphDependency::
    GraphDependency()
{
}

auto GraphDependency::
    setType(GraphDependencyType type)
        -> void
{
    mType = type;
}

auto GraphDependency::
    getType()
        const -> GraphDependencyType
{
    return mType;
}
GraphDependency::
    GraphDependency(GraphDependencyType type)
{
    setType(type);
}

GraphDependency::
    GraphDependency(const dataDependency::Token& token)
    : mType(GraphDependencyType::data)
{
    mTokens.push_back(token);
}

GraphDependency::
    GraphDependency(const std::vector<Neon::set::dataDependency::Token>& tokens)
    : mType(GraphDependencyType::data)
{
    mTokens = tokens;
}

auto GraphDependency::
    getLabel()
        -> std::string
{
    return GraphDependencyTypeUtil::toString(getType());
}

auto GraphDependency::
    addToken(const Neon::set::dataDependency::Token& token) -> void
{
    mTokens.push_back(token);
}



auto GraphDependency::getTokens() const -> const std::vector<Neon::set::dataDependency::Token>&
{
    return mTokens;
}

auto GraphDependency::hasStencilDependency()
    const -> bool
{
    bool isStencil = std::any_of(mTokens.begin(),
                                 mTokens.end(),
                                 [](const auto& token) {
                                     return token.compute() == Neon::Compute::STENCIL;
                                 });
    return isStencil;
}


}  // namespace Neon::set::container
