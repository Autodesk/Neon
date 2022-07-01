
#include <map>
#include "./containers/Axpy.h"
#include "RunTest.h"

namespace Test::sequence {


template <typename Grid,
          typename Type,
          int Cardinality>
auto mapAxpy(Cli::UserData&                                                  userData,
             Neon::domain::tool::testing::TestData<Grid, Type, Cardinality>& testData)
    -> std::vector<Neon::set::Container>
{
    using FieldNames = Neon::domain::tool::testing::FieldNames;

    std::vector<Neon::set::Container> sequence;
    Type const                        alpha = Type(-1);

    auto&         X = testData.getField(FieldNames::X);
    auto&         Y = testData.getField(FieldNames::Y);
    constexpr int cardinality = std::remove_reference_t<decltype(X)>::Cardinality;
    if (X.getCardinality() == 0) {
        NEON_THROW_UNSUPPORTED_OPTION("mapAxpy works only with cardinality != 0");
    }
    if (X.getCardinality() == cardinality) {
        sequence.push_back(Test::containers::axpy<cardinality>(X, alpha, Y));
    } else {
        sequence.push_back(Test::containers::axpy<0>(X, alpha, Y));
    }

    return sequence;
}

template <typename Grid,
          typename Type,
          int Cardinality>
auto mapAxpyGoldenRun(Cli::UserData&                                                  userData,
                      Neon::domain::tool::testing::TestData<Grid, Type, Cardinality>& testData)
    -> void
{  // Golden data
    using FieldNames = Neon::domain::tool::testing::FieldNames;

    auto& X = testData.getIODomain(FieldNames::X);
    auto& Y = testData.getIODomain(FieldNames::Y);

    Type const alpha = Type(-1);

    testData.forEachActiveIODomain([&](const Neon::index_3d& idx,
                                       int                   cardinality,
                                       Type&                 a,
                                       Type&                 b) {
        b += alpha * a;
    },
                                   X, Y);
}
}  // namespace Test::sequence