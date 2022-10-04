#include "nodeContainer.h"


template <typename StaggeredGrid, typename T>
auto Containers<StaggeredGrid, T>::mapOnNodes(Self::NodeField&  inputField,
                                              Self::NodeField&  outputField,
                                              const Self::Type& alpha) -> Neon::set::Container
{
    return inputField.getGrid().getNode(
        "MAP-on-nodes",
        [&](Neon::set::Loader& loader) {
            const auto& inp = loader.load(inputField);
            auto&       out = loader.load(outputField);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Self::NodeField::Node& e) mutable {
                for (int i = 0; i < inp.cardinality(); i++) {
                    out(e, i) = inp(e, i) + alpha;
                }
            };
        });
}

template struct Containers<Neon::domain::internal::experimental::staggeredGrid::StaggeredGrid<Neon::domain::dGrid>, double>;
