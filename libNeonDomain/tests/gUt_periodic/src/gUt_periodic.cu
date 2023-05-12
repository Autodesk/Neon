#include "gtest/gtest.h"

#include "Neon/domain/dGrid.h"
#include "gUt_storage.h"


template <typename Field>
auto stencil(Field& input_field,
             Field& output_field) -> Neon::set::Container
{
    return input_field.grid().container(
        "LFStencil",
        [&](Neon::set::Loader& loader) {
            const auto& inp = loader.load(input_field);
            auto&       out = loader.load(output_field, Neon::Compute::STENCIL);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Field::Cell& e) mutable {
                for (int i = 0; i < inp.cardinality(); i++) {
                    out(e, i) = inp.nghVal(e, {0, 0, -1}, i, -1).value + inp.nghVal(e, {0, 0, 1}, i, -1).value;
                }
            };
        });
}

TEST(Periodic, dGrid)
{
    using GridT = Neon::domain::dense::dGrid;
    using T = float;
    const Neon::index64_3d dim(1, 1, 9);
    const std::vector<int> nGPUs{1, 2, 3};

    const std::vector<int> cardinality{1, 3};

    const std::vector<Neon::set::Backend_t::runtime_et::e> backends{
        Neon::set::Backend_t::runtime_et::e::openmp,
        Neon::set::Backend_t::runtime_et::e::stream};

    const std::vector<Neon::memLayout_et::order_e> layout{
        Neon::memLayout_et::order_e::structOfArrays,
        Neon::memLayout_et::order_e::arrayOfStructs};

    for (const auto& nGPU : nGPUs) {
        for (const auto& card : cardinality) {
            for (const auto& bk : backends) {
                for (const auto& lay : layout) {

                    std::stringstream s;
                    s << " dim " << dim
                              << "ngpu " << nGPU
                              << " cardinality " << card
                              << " backend " << Neon::set::Backend_t::toString(bk)
                              << " layout " << Neon::memOrder_e::toString(lay);

                    NEON_INFO(s.str());

                    Storage<GridT, T> storage(dim, nGPU, card, bk, lay);
                    storage.initConst(-1);

                    storage.Xf.enablePeriodicAlongZ();
                    storage.Yf.enablePeriodicAlongZ();

                    storage.Xf.updateDeviceData(0);

                    storage.Xf.template haloUpdate<Neon::set::Transfer_t::get>(
                        storage.m_backend);

                    auto container = stencil(storage.Xf, storage.Yf);

                    container.run(0);

                    storage.Yf.updateHostData(0);

                    storage.Yf.forEachActive([&](const Neon::index_3d& e,
                                                 const int32_t&        card,
                                                 T&                    val) {
                        EXPECT_NEAR(val, T(2), std::numeric_limits<T>::epsilon());
                    });
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
