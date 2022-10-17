#include <memory.h>

#include <cassert>
#include "Neon/domain/aGrid.h"
#include "Neon/domain/eGrid.h"

#pragma once

template <typename grid_ta, typename T, int card_ta = 0>
class storage_t
{
    using Grid = grid_ta;
    using Field = typename Grid::template Field<T, card_ta>;

   public:
    // Neon::set::DevSet m_devSet;
    Grid             m_grid;
    Neon::index64_3d m_size3d;
    int              m_cardinality;

   public:
    Field                Xf, Yf, Zf;
    std::shared_ptr<T[]> Xd, Yd, Zd;

    // TODO remove  Xd, Yd, Zd, and use directly Xnd, Ynd, Znd
    Neon::IODense<T> Xnd, Ynd, Znd;
    Neon::Backend    backend;


   private:
    auto getNewDenseField(Neon::index64_3d      size3d,
                          int                   cardinality,
                          std::shared_ptr<T[]>& d,
                          Neon::IODense<T>&     nd)
        -> void
    {
        nd = Neon::IODense<T>(size3d.template newType<typename Neon::IODense<T>::Index>(),
                              cardinality);
        d = nd.getSharedPtr();
    }

    void setLinearly(int offset, std::shared_ptr<T[]>& dense)
    {
#ifdef _MSC_VER
#else
#pragma omp parallel for collapse(2)
#endif
        for (int64_t card = 0; card < m_cardinality; card++) {
            for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                dense.get()[i + m_size3d.rMul() * card] = (100 * offset) * (i + 1) + card;
            }
        }
    }

    void setConstant(int /*offset*/, std::shared_ptr<T[]>& dense, int constVal)
    {
#ifdef _MSC_VER
#else
#pragma omp parallel for collapse(2)
#endif
        for (int64_t card = 0; card < m_cardinality; card++) {
            for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                dense.get()[i + m_size3d.rMul() * card] = constVal + card;
            }
        }
    }

    void setLinearly(int offset, std::shared_ptr<T[]>& dense, int targetCard)
    {
#ifdef _MSC_VER
#else
#pragma omp parallel for collapse(1)
#endif
        for (int64_t card = 0; card < m_cardinality; card++) {
            if (card == targetCard) {
                for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                    dense.get()[i + m_size3d.rMul() * card] = (100 * offset) * (i + 1) + card;
                }
            } else {
                for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                    dense.get()[i + m_size3d.rMul() * card] = 0;
                }
            }
        }
    }

    void setConstant(int /*offset*/, std::shared_ptr<T[]>& dense, int targetCard, int constVal)
    {
#ifdef _MSC_VER
#else
#pragma omp parallel for collapse(1)
#endif
        for (int64_t card = 0; card < m_cardinality; card++) {
            if (card == targetCard) {
                for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                    dense.get()[i + m_size3d.rMul() * card] = constVal + card;
                }
            } else {
                for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                    dense.get()[i + m_size3d.rMul() * card] = 0;
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
    void ioToVti(const std::string fname)
    {
        Xf.updateIO(0);
        Yf.updateIO(0);
        Zf.updateIO(0);
        Xf.getGrid().getBackend().sync(0);

        auto Xd_val = [&](const Neon::int64_3d& index3d, int card) {
            return double(Xd.get()[index3d.x + m_size3d.rMul() * card]);
        };
        auto Yd_val = [&](const Neon::int64_3d& index3d, int card) {
            return double(Yd.get()[index3d.x + m_size3d.rMul() * card]);
        };
        auto Zd_val = [&](const Neon::int64_3d& index3d, int card) {
            return double(Zd.get()[index3d.x + m_size3d.rMul() * card]);
        };
        Neon::int64_3d l(m_size3d.rMul(), 1, 1);
        Neon::IoToVTK  ioToVTK(fname + "_field.vti",
                              l,
                              1.0,
                              0.0, Neon::IoFileType::ASCII);

        ioToVTK.addField(Xd_val, m_cardinality, "Xd", Neon::ioToVTKns::node);
        ioToVTK.addField(Yd_val, m_cardinality, "Yd", Neon::ioToVTKns::node);
        ioToVTK.addField(Zd_val, m_cardinality, "Zd", Neon::ioToVTKns::node);

        ioToVTK.flush();
    }

   public:
    void initLinearly()
    {
        setLinearly(1, Xd);
        setLinearly(2, Yd);
        setLinearly(3, Zd);

        loadField(Xnd, Xf);
        loadField(Ynd, Yf);
        loadField(Znd, Zf);
    }

    void initConst(int a = 1, int b = 2, int c = 3)
    {
        setConstant(1, Xd, a);
        setConstant(2, Yd, b);
        setConstant(3, Zd, c);

        loadField(Xd, Xf);
        loadField(Yd, Yf);
        loadField(Zd, Zf);
    }

    void initLinearly(int targetCard)
    {
        setLinearly(1, Xd, targetCard);
        setLinearly(2, Yd, targetCard);
        setLinearly(3, Zd, targetCard);

        loadField(Xd, Xf);
        loadField(Yd, Yf);
        loadField(Zd, Zf);
    }

    bool compare(std::shared_ptr<T[]>& a, Field& bField)
    {
        int same = 0;
        backend.sync();
        bField.updateIO(0);
        bField.getGrid().getBackend().sync();
        std::shared_ptr<T[]> b = bField.ioToDense().getSharedPtr();

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

    void runKernel(std::shared_ptr<T[]>& a, std::shared_ptr<T[]>& b, std::shared_ptr<T[]>& c, T val)
    {
#ifdef _MSC_VER
#else
#pragma omp parallel for collapse(2)
#endif
        for (int64_t card = 0; card < m_cardinality; card++) {
            for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                c.get()[i + m_size3d.rMul() * card] = a.get()[i + m_size3d.rMul() * card] + b.get()[i + m_size3d.rMul() * card] + val;
            }
        }
    }

    void lamdaTest(std::shared_ptr<T[]>& a, std::shared_ptr<T[]>& c)
    {
#ifdef _MSC_VER
#else
#pragma omp parallel for collapse(2)
#endif
        for (int64_t card = 0; card < m_cardinality; card++) {
            for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                c.get()[i + m_size3d.rMul() * card] = a.get()[i + m_size3d.rMul() * card] + 33;
            }
        }
    }

    static void gridInit(Neon::index64_3d dim, storage_t<Neon::domain::aGrid, T>& storage, const Neon::Backend& backendConfig)
    {
        storage.m_size3d = {1, 1, 1};
        storage.m_size3d.x = dim.template rMulTyped<int64_t>();

        auto    lengths = storage.backend.devSet().template newDataSet<Neon::domain::aGrid::Count>(storage.m_size3d.x / storage.backend.devSet().setCardinality());
        int64_t sumTmp = 0;
        for (int i = 0; i < storage.m_size3d.x % storage.backend.devSet().setCardinality(); i++) {
            lengths[i]++;
        }
        for (int i = 0; i < storage.backend.devSet().setCardinality(); i++) {
            sumTmp += lengths[i];
        }
        assert(sumTmp == storage.m_size3d.x);
        storage.m_grid = Neon::domain::aGrid(storage.backend.devSet(), lengths);

        storage.Xf = storage.m_grid.template newField<T>({backendConfig, Neon::DataUse::IO_COMPUTE}, storage.m_cardinality);
        storage.Yf = storage.m_grid.template newField<T>({backendConfig, Neon::DataUse::IO_COMPUTE}, storage.m_cardinality);
        storage.Zf = storage.m_grid.template newField<T>({backendConfig, Neon::DataUse::IO_COMPUTE}, storage.m_cardinality);
    }

    static void gridInit(Neon::index64_3d                                             dim,
                         storage_t<Neon::domain::internal::eGrid::eGrid, T, card_ta>& storage,
                         const Neon::Backend&,
                         const Neon::MemSetOptions_t& memSetOptions)
    {
        storage.m_size3d = dim;

        storage.m_grid = Neon::domain::internal::eGrid::eGrid(
            storage.backend,
            dim.template newType<int32_t>(),
            [&](const Neon::index_3d& /*idx*/) -> bool {
                return true;
            },
            Neon::domain::Stencil::s7_Laplace_t(),
            false);

        T outsideVal = 0;
        //        storage.Xf = storage.m_grid.template newField<T, card_ta>({backendConfig, Neon::DataUse::IO_COMPUTE}, Neon::domain::internal::eGrid::haloStatus_et::OFF, storage.m_cardinality, outsideVal);
        //        storage.Yf = storage.m_grid.template newField<T, card_ta>({backendConfig, Neon::DataUse::IO_COMPUTE}, Neon::domain::internal::eGrid::haloStatus_et::OFF, storage.m_cardinality, outsideVal);
        //        storage.Zf = storage.m_grid.template newField<T, card_ta>({backendConfig, Neon::DataUse::IO_COMPUTE}, Neon::domain::internal::eGrid::haloStatus_et::OFF, storage.m_cardinality, outsideVal);


        //        Neon::MemSetOptions_t memSetOptions;
        //        memSetOptions.order() = order == Neon::memLayout_et::order_e::structOfArrays ? Neon::memOrder_e::structOfArrays : Neon::memOrder_e::arrayOfStructs;
        //        memSetOptions.alignment() = Neon::memAlignment_e::PAGE;
        //        memSetOptions.padding() = Neon::memPadding_e::ON;
        // memSetOptions.alignment() = Neon::memAlignment_e::PAGE;

        storage.Xf = storage.m_grid.template newField<T, card_ta>("Xf", storage.m_cardinality, outsideVal, Neon::DataUse::IO_COMPUTE);
        storage.Yf = storage.m_grid.template newField<T, card_ta>("Yf", storage.m_cardinality, outsideVal, Neon::DataUse::IO_COMPUTE);
        storage.Zf = storage.m_grid.template newField<T, card_ta>("Zf", storage.m_cardinality, outsideVal, Neon::DataUse::IO_COMPUTE);
    }

    storage_t(Neon::index64_3d            dim,
              int                         nGPUs,
              int                         cardinality,
              const Neon::Runtime&        backendType,
              const Neon::MemSetOptions_t memSetOptions)
    {
        m_cardinality = cardinality;
        std::vector<int> gpusIds(nGPUs, 0);
        backend = Neon::Backend(gpusIds, backendType);

        gridInit(dim, *this, backend, memSetOptions);

        getNewDenseField(m_size3d, m_cardinality, Xd, Xnd);
        getNewDenseField(m_size3d, m_cardinality, Yd, Ynd);
        getNewDenseField(m_size3d, m_cardinality, Zd, Znd);
    }
};
