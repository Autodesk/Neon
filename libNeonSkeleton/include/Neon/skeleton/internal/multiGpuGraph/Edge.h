#pragma once
#include "Neon/set/dependencyTools/DataParsing.h"
#include "Neon/skeleton/internal/dependencyTools/Alias.h"

namespace Neon {
namespace skeleton {
namespace internal {

/**
 * Information stored for each Edge in the user graph
 */
struct Edge
{
    struct Info
    {
        Info(DataToken_t    t,
               Dependencies_e d,
               bool           haloUp = false)
            : token(t), dependency(d), haloUpdate(haloUp)
        {
            if (haloUp && token.compute() != Neon::Compute::STENCIL) {
                NEON_THROW_UNSUPPORTED_OPTION("");
            }
        }

        DataToken_t token;
        // This flag is true when a stencil dependency does need
        // halo update as the previous map is running on internals
        bool           flag_discardStencilDep = {false};
        Dependencies_e dependency;
        bool           haloUpdate = false;

        /**
         * Returns true if this is a stencil edge
         * @return
         */
        bool isStencil() const
        {
            return (token.compute() == Neon::Compute::STENCIL) && !flag_discardStencilDep;
        }

        /**
         * Returns true if this is halo edge
         * @return
         */
        bool isHu() const
        {
            return haloUpdate;
        }

        auto toString() const -> std::string
        {
            using namespace set::internal::dependencyTools;
            if (!haloUpdate) {
                //                return " UID " + std::to_string(token.uid()) +
                return "- Type: " + Dependencies_et::toString(dependency) + "\\l";  //"- UID: " + std::to_string(token.uid()) + "\\l";
                //                       "\\l-Operation: " + Access_et::toString(token.access()) +
                //                       "\\l-Pattern:  " + Compute_et::toString(token.compute())+

            } else {
                //                return " UID " + std::to_string(token.uid()) +
                return "- Type: " + Dependencies_et::toString(dependency) + " HU\\l";
                //                       "\\l-Operation: " + Access_et::toString(token.access()) +
                //                       "\\l-Pattern: " + Compute_et::toString(token.compute())+
                //                       " HU\\l";
            }
        }
    };

    size_t              m_edgeId;
    std::vector<Info> m_dependencies;
    bool                m_isSchedulingEdge = false;
    /**
     * Private constructor.
     * Only the container method can be used to create Edge elements.
     *
     * @param edgeId
     * @param op
     * @param uid
     */
    Edge(size_t             edgeId,
           const DataToken_t& dataToken,
           Dependencies_e     dependency,
           bool               haloUp = false);

    Edge(size_t edgeId,
           bool   isSchedulingEdge);

   public:
    Edge() = default;

    /**
     * Factory method to create an Edge object.
     * This methos is not thread safe
     *
     * @param op
     * @return
     */
    static auto
    factory(const DataToken_t& dataToken,
            Dependencies_e     dependency,
            bool               haloUp = false) -> Edge;

    static auto
    factorySchedulingEdge() -> Edge;

    auto clone() const -> Edge;

    auto edgeId() const -> size_t;

    auto
    nDependencies()
        const -> size_t;

    auto
    info()
        const -> const std::vector<Info>&;

    auto
    infoMutable()
        -> std::vector<Info>&;


    auto
    append(const DataToken_t& dataToken,
           Dependencies_e     dType) -> void;

    auto
    toString()
        const -> std::string;
};

}  // namespace internal
}  // namespace skeleton
}  // namespace Neon