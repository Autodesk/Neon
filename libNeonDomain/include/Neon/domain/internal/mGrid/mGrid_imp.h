#include "Neon/domain/internal/mGrid/mGrid.h"

namespace Neon::domain::internal::mGrid {

template <typename T, int C>
auto mGrid::newField(const std::string          name,
                     int                        cardinality,
                     T                          inactiveValue,
                     Neon::DataUse              dataUse,
                     const Neon::MemoryOptions& memoryOptions) const -> Field<T, C>
{
    mField<T, C> field(name, *this, cardinality, inactiveValue, dataUse, memoryOptions, Neon::domain::haloStatus_et::ON);

    return field;
}


template <typename LoadingLambda>
auto mGrid::getContainer(const std::string& name,
                         int                level,
                         index_3d           blockSize,
                         size_t             sharedMem,
                         LoadingLambda      lambda) const -> Neon::set::Container
{


    Neon::set::Container kContainer = mData->grids[level].getContainer(name, blockSizem sharedMem, lambda);

    return kContainer;
}

template <typename LoadingLambda>
auto mGrid::getContainer(const std::string& name,
                         int                level,
                         LoadingLambda      lambda) const -> Neon::set::Container
{
    Neon::set::Container kContainer = mData->grids[level].getContainer(name, lambda);

    return kContainer;
}

/*template <typename T>
auto mGrid::newPatternScalar() const -> Neon::template PatternScalar<T>
{
    // TODO this sets the numBlocks for only Standard dataView.
    auto pattern = Neon::PatternScalar<T>(getBackend(), Neon::sys::patterns::Engine::CUB);
    for (SetIdx id = 0; id < mData->mNumBlocks[0].cardinality(); id++) {
        pattern.getBlasSet(Neon::DataView::STANDARD).getBlas(id.idx()).setNumBlocks(uint32_t(mData->mNumBlocks[0][id]));
    }
    return pattern;
}


template <typename T>
auto mGrid::dot(const std::string&               name,
                Field<T>&                        input1,
                Field<T>&                        input2,
                Neon::template PatternScalar<T>& scalar,
                const int                        level) const -> Neon::set::Container
{
    return Neon::set::Container::factoryOldManaged(
        name,
        Neon::set::internal::ContainerAPI::DataViewSupport::on,
        *this, [&](Neon::set::Loader& loader) {
            loader.load(input1);
            if (input1.getUid() != input2.getUid()) {
                loader.load(input2);
            }

            return [&](int streamIdx, Neon::DataView dataView) mutable -> void {
                if (dataView != Neon::DataView::STANDARD) {
                    NeonException exc("mGrid");
                    exc << "Reduction operation on mGrid works only on standard dataview";
                    exc << "Input dataview is" << Neon::DataViewUtil::toString(dataView);
                    NEON_THROW(exc);
                }

                if (dataView != Neon::DataView::STANDARD && getBackend().devSet().setCardinality() == 1) {
                    NeonException exc("mGrid");
                    exc << "Reduction operation can only run on standard data view when the number of partitions/GPUs is 1";
                    NEON_THROW(exc);
                }

                if (getBackend().devType() == Neon::DeviceType::CUDA) {
                    scalar.setStream(streamIdx, dataView);

                    // calc dot product and store results on device
                    input1.dot(scalar.getBlasSet(dataView),
                               input2,
                               scalar.getTempMemory(dataView, Neon::DeviceType::CUDA),
                               dataView, level);

                    // move to results to host
                    scalar.getTempMemory(dataView,
                                         Neon::DeviceType::CPU)
                        .template updateFrom<Neon::run_et::et::async>(
                            scalar.getBlasSet(dataView).getStream(),
                            scalar.getTempMemory(dataView, Neon::DeviceType::CUDA));

                    // sync
                    scalar.getBlasSet(dataView).getStream().sync();

                    // read the results
                    scalar() = scalar.getTempMemory(dataView, Neon::DeviceType::CPU).elRef(0, 0, 0);
                } else {

                    scalar() = 0;
                    input1.template forEachActiveCell<Neon::computeMode_t::computeMode_e::seq>(
                        [&](const Neon::index_3d& idx,
                            const int&            cardinality,
                            T&                    in1) {
                            scalar() += in1 * input2(idx, cardinality);
                        });
                }
            };
        });
}*/

/*template <typename T>
auto mGrid::norm2(const std::string&               name,
                  Field<T>&                        input,
                  Neon::template PatternScalar<T>& scalar,
                  const int                        level) const -> Neon::set::Container
{
    return Neon::set::Container::factoryOldManaged(
        name,
        Neon::set::internal::ContainerAPI::DataViewSupport::on,
        *this, [&](Neon::set::Loader& loader) {
            loader.load(input);


            return [&](int streamIdx, Neon::DataView dataView) mutable -> void {
                if (dataView != Neon::DataView::STANDARD) {
                    NeonException exc("mGrid");
                    exc << "Reduction operation on mGrid works only on standard dataview";
                    exc << "Input dataview is" << Neon::DataViewUtil::toString(dataView);
                    NEON_THROW(exc);
                }

                if (dataView != Neon::DataView::STANDARD && getBackend().devSet().setCardinality() == 1) {
                    NeonException exc("mGrid");
                    exc << "Reduction operation can only run on standard data view when the number of partitions/GPUs is 1";
                    NEON_THROW(exc);
                }

                if (getBackend().devType() == Neon::DeviceType::CUDA) {
                    scalar.setStream(streamIdx, dataView);

                    // calc dot product and store results on device
                    input.norm2(scalar.getBlasSet(dataView),
                                scalar.getTempMemory(dataView, Neon::DeviceType::CUDA),
                                dataView, level);

                    // move to results to host
                    scalar.getTempMemory(dataView,
                                         Neon::DeviceType::CPU)
                        .template updateFrom<Neon::run_et::et::async>(
                            scalar.getBlasSet(dataView).getStream(),
                            scalar.getTempMemory(dataView, Neon::DeviceType::CUDA));

                    // sync
                    scalar.getBlasSet(dataView).getStream().sync();

                    // read the results
                    scalar() = scalar.getTempMemory(dataView, Neon::DeviceType::CPU).elRef(0, 0, 0);
                } else {

                    scalar() = 0;
                    input.template forEachActiveCell<Neon::computeMode_t::computeMode_e::seq>(
                        [&]([[maybe_unused]] const Neon::index_3d& idx,
                            [[maybe_unused]] const int&            cardinality,
                            T&                                     in) {
                            scalar() += in * in;
                        });
                }
                scalar() = std::sqrt(scalar());
            };
        });
}*/
}  // namespace Neon::domain::internal::mGrid