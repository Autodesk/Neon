#include <memory.h>

#include <cassert>
#include "Neon/Report.h"

#include "Neon/domain/aGrid.h"
#include "Neon/domain/eGrid.h"
#include "Neon/skeleton/Skeleton.h"
#include "sPt_geometry.h"

#pragma once


template <typename G, typename T>
class storage_t
{
    using Grid = G;
    using Field = typename Grid::template Field<T>;

   public:
    Grid             m_grid;
    Neon::index64_3d m_size3d;
    int              m_cardinality;
    geometry_t       m_geo;
    Neon::Backend    m_backend;

   public:
    Field                Xf, Yf, Zf;
    std::shared_ptr<T[]> Xd, Yd, Zd;
    // TODO remove  Xd, Yd, Zd, and use directly Xnd, Ynd, Znd
    Neon::IODense<T>      Xnd, Ynd, Znd;
    Neon::domain::Stencil m_stencil = Neon::domain::Stencil::s7_Laplace_t();

   private:
    std::shared_ptr<T[]> getNewDenseField(Neon::index64_3d size3d, int cardinality)
    {
        size_t               size = size3d.template rMulTyped<size_t>();
        std::shared_ptr<T[]> d(new T[size * cardinality],
                               [](T* i) {
                                   delete[] i;  // Custom delete
                               });
        return d;
    }

    void setLinearly(int offset, std::shared_ptr<T[]>& dense, T outsideDomain = 0, int targetCard = -1)
    {
#pragma omp parallel for collapse(2)
        for (int64_t card = 0; card < m_cardinality; card++) {
            for (int z = 0; z < m_size3d.z; z++) {
                for (int y = 0; y < m_size3d.y; y++) {
                    for (int x = 0; x < m_size3d.x; x++) {
                        auto         xyz = index_3d(x, y, z);
                        const size_t i = xyz.mPitch(m_size3d);
                        T            val = outsideDomain;
                        if (m_geo(xyz)) {
                            if (card == targetCard || targetCard == -1) {
                                val = T((100000000 * offset) + 1000000 * card + 10000 * z + 100 * y + x);
                            } else {
                                val = 0;
                            }
                        }
                        dense.get()[i + m_size3d.rMul() * card] = val;
                    }
                }
            }
        }
    }

    void loadField(const Neon::IODense<T>& dense, Field& field)
    {
        field.ioFromDense(dense);
        field.updateCompute(0);
        field.getGrid().getBackend().sync(0);
    }

   public:
    void axpy_f([[maybe_unused]] std::shared_ptr<T[]>& X, [[maybe_unused]] T a, [[maybe_unused]] std::shared_ptr<T[]>& Y)
    {

        for (int64_t card = 0; card < m_cardinality; card++) {
            for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                Y.get()[i + m_size3d.rMul() * card] += a * X.get()[i + m_size3d.rMul() * card];
            }
        }
    }

    auto laplacianFilter_f(const std::shared_ptr<T[]>&                   in,
                           [[maybe_unused]] std::shared_ptr<T[]>&        out,
                           [[maybe_unused]] std::vector<index_3d> const& directions) -> void
    {
        const int cardinality = 0;
        const T   outsideDomain = 0;
        //#pragma omp parallel for collapse(2)
        for (int z = 0; z < m_size3d.z; z++) {
            for (int y = 0; y < m_size3d.y; y++) {
                for (int x = 0; x < m_size3d.x; x++) {
                    index64_3d const xyz(x, y, z);
                    const size_t     i = xyz.mPitch(m_size3d);
                    T                partial = 0;
                    for (auto const& direction : directions) {
                        index64_3d xyzPlusOff = xyz + direction.newType<int64_t>();

                        const size_t ioff = xyzPlusOff.mPitch(m_size3d);

                        auto isInDomain = [&](index64_3d p) {
                            auto res = true;
                            res = res && p.x >= 0;
                            res = res && p.y >= 0;
                            res = res && p.z >= 0;
                            res = res && p.x < m_size3d.x;
                            res = res && p.y < m_size3d.y;
                            res = res && p.z < m_size3d.z;
                            return m_geo(xyzPlusOff.newType<int>()) && res;
                        };

                        T val = outsideDomain;
                        if (isInDomain(xyzPlusOff)) {
                            size_t jump = ioff + m_size3d.rMul() * cardinality;
                            val = in.get()[jump];
                        }
                        partial += val;
                    }
                    if (m_geo(xyz.newType<int>())) {
                        [[maybe_unused]] T val = -partial +
                                                 6 * in.get()[i + m_size3d.rMul() * cardinality];
                        out.get()[i + m_size3d.rMul() * cardinality] = val;
                    }
                }
            }
        }
    }


    void ioToVti(const std::string fname)
    {
        Xf.updateIO(Xf.grid().devSet().defaultStreamSet());
        Yf.updateIO(Yf.grid().devSet().defaultStreamSet());

        Xf.grid().devSet().defaultStreamSet().sync();

        auto Xd_val = [&](const Neon::int64_3d& index3d, int card) {
            return double(Xd.get()[index3d.mPitch(m_size3d) + m_size3d.rMul() * card]);
        };
        auto Yd_val = [&](const Neon::int64_3d& index3d, int card) {
            return double(Yd.get()[index3d.mPitch(m_size3d) + m_size3d.rMul() * card]);
        };

        Neon::int64_3d l = m_size3d;
        Neon::ioToVTI<int64_t>({{Xd_val, m_cardinality, "Xd", true, Neon::IoFileType::ASCII},
                                {Yd_val, m_cardinality, "Yd", true, Neon::IoFileType::ASCII}},
                               fname + ".dense.vti", l, l - 1);
        m_grid.template ioVtk<T>({{&Xf, "Xf", true}, {&Yf, "Yf", true}}, fname + ".grid.vti");
    }

   public:
    void
    initLinearly()
    {
        setLinearly(1, Xd);
        setLinearly(2, Yd);


        loadField(Xnd, Xf);
        loadField(Ynd, Yf);

        bool a = compare(Xd, Xf);
        bool b = compare(Yd, Yf);

        if (!(a && b)) {
            NEON_THROW_UNSUPPORTED_OPERATION();
        }
    }

    void initLinearly(int targetCard)
    {
        setLinearly(1, Xd, targetCard);
        setLinearly(2, Yd, targetCard);

        loadField(Xnd, Xf);
        loadField(Ynd, Yf);
    }

    bool compare(std::shared_ptr<T[]>& a, Field& bField)
    {

        int same = 0;
        m_backend.sync();
        bField.updateIO(0);
        m_backend.sync();

        std::shared_ptr<T[]> b;
        b = bField.ioToDense().getSharedPtr();


#ifdef _MSC_VER
#else
#pragma omp parallel for collapse(2) reduction(+ \
                                               : same)
#endif
        for (int64_t card = 0; card < m_cardinality; card++) {
            for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                if (a.get()[i + m_size3d.rMul() * card] != b.get()[i + m_size3d.rMul() * card]) {
                    same += 1;
                }
            }
        }
        return same == 0;
    }

    static void gridInit(Neon::index64_3d dim,
                         geometry_t /*geo*/,
                         storage_t<Neon::aGrid, T>& storage,
                         const Neon::Backend&               backendConfig)
    {
        storage.m_size3d = {1, 1, 1};
        storage.m_size3d.x = dim.template rMulTyped<int64_t>();

        auto    lengths = storage.m_devSet.template newDataSet<Neon::aGrid::Count>(storage.m_size3d.x / storage.m_devSet.setCardinality());
        int64_t sumTmp = 0;
        for (int i = 0; i < storage.m_size3d.x % storage.m_devSet.setCardinality(); i++) {
            lengths[i]++;
        }
        for (int i = 0; i < storage.m_devSet.setCardinality(); i++) {
            sumTmp += lengths[i];
        }
        assert(sumTmp == storage.m_size3d.x);
        storage.m_grid = Neon::aGrid(storage.m_devSet, lengths);

        storage.Xf = storage.m_grid.template newField<T>({backendConfig, Neon::DataUse::HOST_DEVICE}, storage.m_cardinality);
        storage.Yf = storage.m_grid.template newField<T>({backendConfig, Neon::DataUse::HOST_DEVICE}, storage.m_cardinality);
        storage.Zf = storage.m_grid.template newField<T>({backendConfig, Neon::DataUse::HOST_DEVICE}, storage.m_cardinality);
    }

    void gridInit(Neon::index64_3d     dim,
                  geometry_t           geo,
                  const Neon::Backend& backendConfig)
    {
        m_size3d = dim;

        m_grid = Grid(backendConfig,
                      dim.template newType<int32_t>(), geo,
                      m_stencil);
        T outsideDomainVal = 0;

        Xf = m_grid.template newField<T>("Xf", m_cardinality, outsideDomainVal, Neon::DataUse::HOST_DEVICE);
        Yf = m_grid.template newField<T>("Yf", m_cardinality, outsideDomainVal, Neon::DataUse::HOST_DEVICE);
        Zf = m_grid.template newField<T>("Zf", m_cardinality, outsideDomainVal, Neon::DataUse::HOST_DEVICE);
    }


    storage_t(Neon::index64_3d dim,
              geometry_t       geo,
              int /*nGPUs*/,
              int                  cardinality,
              const Neon::Backend& backendConfig)
    {
        m_cardinality = cardinality;
        m_backend = backendConfig;
        m_geo = geo;

        gridInit(dim, geo, backendConfig);

        Xd = getNewDenseField(m_size3d, cardinality);
        Yd = getNewDenseField(m_size3d, cardinality);
    }
};

namespace {

std::array<std::string, 3> OccStr{
    "NO_OCC",
    "OCC",
    "EXTENDED_OCC",
};

std::array<std::string, 2> DataTypeStr{
    "INT64_TYPE",
    "DOUBLE_TYPE",
};


enum DataType
{
    INT64_TYPE,
    DOUBLE_TYPE,
};


[[maybe_unused]] auto DataTypeStr2Val(std::string opt) -> DataType
{
    if (opt == DataTypeStr[INT64_TYPE])
        return INT64_TYPE;
    if (opt == DataTypeStr[DOUBLE_TYPE])
        return DOUBLE_TYPE;
    opt = opt + "_TYPE";
    if (opt == DataTypeStr[INT64_TYPE])
        return INT64_TYPE;
    if (opt == DataTypeStr[DOUBLE_TYPE])
        return DOUBLE_TYPE;
    NEON_THROW_UNSUPPORTED_OPTION();
}


struct TestConfigurations
{
    Neon::index64_3d        m_dim{0};
    geometry_t              m_geo{topologies_e::FullDomain, Neon::index64_3d{0, 0, 0}};
    int                     m_nIterations{0};
    int                     m_nRepetitions{0};
    int                     m_nGPUs{0};
    int                     m_maxnGPUs{0};
    Neon::skeleton::Occ     m_optSkelOCC;
    Neon::set::TransferMode m_optSkelTransfer;

    Neon::Backend  m_backend;
    bool           m_compare{false};
    Neon::Timer_ms m_timer;
    std::string    m_fnamePrefix{"NO_NAME"};
    DataType       m_dataType = {INT64_TYPE};

    auto getSklOpt() -> Neon::skeleton::Options
    {
        Neon::skeleton::Options opt(m_optSkelOCC, m_optSkelTransfer);
        return opt;
    }

    auto storeInforInReport(Neon::Report& report, int maxnGPUs)
    {
        report.addMember("dimx", m_dim.x);
        report.addMember("dimy", m_dim.y);
        report.addMember("dimz", m_dim.z);

        report.addMember("cardinality", 1);
        report.addMember("numGPUs", m_nGPUs);
        report.addMember("maxAvailableGPUs", maxnGPUs);

        report.addMember("optimization", Neon::skeleton::OccUtils::toString(m_optSkelOCC));
        report.addMember("transfer", Neon::set::TransferModeUtils::toString(m_optSkelTransfer));

        report.addMember("dataType", DataTypeStr[m_dataType]);

        report.addMember("nIterations", m_nIterations);
        report.addMember("nRepetitions", m_nRepetitions);
        report.addMember("fnamePrefix", m_fnamePrefix);
        report.addMember("compare", m_compare);
    }


    auto toString() -> std::string
    {
        std::stringstream s;
        int               cardinality = 1;
        auto              ptSkelOCC =  Neon::skeleton::OccUtils::toString(m_optSkelOCC);
        auto              ptSkelTransfer = Neon::set::TransferModeUtils::toString(m_optSkelTransfer);

        auto dataType = DataTypeStr[this->m_dataType];

#define MY_STREAM_PRINT(X) s << #X \
                             << " " << X << std::endl;
        MY_STREAM_PRINT(m_dim);
        MY_STREAM_PRINT(cardinality);
        MY_STREAM_PRINT(m_nGPUs);
        MY_STREAM_PRINT(m_maxnGPUs);
        MY_STREAM_PRINT(ptSkelOCC);
        MY_STREAM_PRINT(ptSkelTransfer);
        MY_STREAM_PRINT(dataType);
        MY_STREAM_PRINT(m_nIterations);
        MY_STREAM_PRINT(m_nRepetitions);
        MY_STREAM_PRINT(m_fnamePrefix);
        MY_STREAM_PRINT(m_compare);

#undef MY_STREAM_PRINT
        return s.str();
    }
};
}  // namespace