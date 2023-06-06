#include <memory.h>

#include <cassert>
#include "Neon/domain/aGrid.h"
#include "Neon/domain/bGrid.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"

#pragma once

template <typename grid_ta, typename T>
class storage_t
{
    using Grid = grid_ta;
    using Field = typename Grid::template Field<T>;

   public:
    Grid             m_grid;
    Neon::index64_3d m_size3d;
    int              m_cardinality;

   public:
    Field                Xf, Yf, Zf;
    std::shared_ptr<T[]> Xd, Yd, Zd;

    // TODO remove  Xd, Yd, Zd, and use directly Xnd, Ynd, Znd
    Neon::IODense<T> Xnd, Ynd, Znd;
    Neon::Backend    m_backend;


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

    void setConsant(std::shared_ptr<T[]>& dense, int constVal)
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

    void setConsant(std::shared_ptr<T[]>& dense, int targetCard, int constVal)
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
        field.updateDeviceData(0);
        field.getGrid().getBackend().sync(0);
    }

   public:
    void ioToVti(const std::string fname)
    {
        Xf.updateHostData(Xf.grid().devSet().defaultStreamSet());
        Yf.updateHostData(Xf.grid().devSet().defaultStreamSet());
        Zf.updateHostData(Xf.grid().devSet().defaultStreamSet());
        Xf.grid().devSet().defaultStreamSet().sync();

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
        Neon::ioToVTI<int64_t>({{Xd_val, m_cardinality, "Xd", true, Neon::IoFileType::ASCII},
                                {Yd_val, m_cardinality, "Yd", true, Neon::IoFileType::ASCII},
                                {Zd_val, m_cardinality, "Zd", true, Neon::IoFileType::ASCII}},
                               fname + "_dense.vti", l, l - 1);
        m_grid.template ioVtk<T>({{&Xf, "Xa", true}, {&Yf, "Ya", true}, {&Zf, "Za", true}}, fname + "_field.vti");
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
        setConsant(Xd, a);
        setConsant(Yd, b);
        setConsant(Zd, c);

        loadField(Xnd, Xf);
        loadField(Ynd, Yf);
        loadField(Znd, Zf);
    }

    void initLinearly(int targetCard)
    {
        setLinearly(1, Xd, targetCard);
        setLinearly(2, Yd, targetCard);
        setLinearly(3, Zd, targetCard);

        loadField(Xnd, Xf);
        loadField(Ynd, Yf);
        loadField(Znd, Zf);
    }

    bool compare(std::shared_ptr<T[]>& a, Field& bField)
    {
        int same = 0;
        m_backend.sync();

        bField.updateHostData(0);
        m_backend.sync();

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

    void copy(std::shared_ptr<T[]>& a, std::shared_ptr<T[]>& c)
    {
#ifdef _MSC_VER
#else
#pragma omp parallel for collapse(2)
#endif
        for (int64_t card = 0; card < m_cardinality; card++) {
            for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                c.get()[i + m_size3d.rMul() * card] = a.get()[i + m_size3d.rMul() * card];
            }
        }
    }

    void sum(std::shared_ptr<T[]>& a, std::shared_ptr<T[]>& b, std::shared_ptr<T[]>& c)
    {
#ifdef _MSC_VER
#else
#pragma omp parallel for collapse(2)
#endif
        for (int64_t card = 0; card < m_cardinality; card++) {
            for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                c.get()[i + m_size3d.rMul() * card] = a.get()[i + m_size3d.rMul() * card] + b.get()[i + m_size3d.rMul() * card];
            }
        }
    }

    void axpy(T a, std::shared_ptr<T[]>& X, std::shared_ptr<T[]>& Y)
    {
#ifdef _MSC_VER
#else
#pragma omp parallel for collapse(2)
#endif
        for (int64_t card = 0; card < m_cardinality; card++) {
            for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                Y.get()[i + m_size3d.rMul() * card] += a * X.get()[i + m_size3d.rMul() * card];
            }
        }
    }

    void axpy(T a, std::shared_ptr<T[]>& X, std::shared_ptr<T[]>& Y, T a1, std::shared_ptr<T[]>& X1, std::shared_ptr<T[]>& Y1)
    {
#ifdef _MSC_VER
#else
#pragma omp parallel for collapse(2)
#endif
        for (int64_t card = 0; card < m_cardinality; card++) {
            for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                Y.get()[i + m_size3d.rMul() * card] += a * X.get()[i + m_size3d.rMul() * card];
                Y1.get()[i + m_size3d.rMul() * card] += a1 * X1.get()[i + m_size3d.rMul() * card];
            }
        }
    }

    void xpay(std::shared_ptr<T[]>& X, T a, std::shared_ptr<T[]>& Y)
    {
#ifdef _MSC_VER
#else
#pragma omp parallel for collapse(2)
#endif
        for (int64_t card = 0; card < m_cardinality; card++) {
            for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                Y.get()[i + m_size3d.rMul() * card] = X.get()[i + m_size3d.rMul() * card] + a * Y.get()[i + m_size3d.rMul() * card];
            }
        }
    }

    T rMaxNorm(std::shared_ptr<T[]>& a, std::shared_ptr<T[]>& b)
    {
        T final = 0;
#ifdef _MSC_VER
#else
#pragma omp parallel for collapse(1) reduction(max \
                                               : final)
#endif
        for (int64_t i = 0; i < m_size3d.rMul(); i++) {
            T cardPartial = 0;
            for (int64_t card = 0; card < m_cardinality; card++) {
                const auto aVal = a.get()[i + m_size3d.rMul() * card];
                const auto bVal = b.get()[i + m_size3d.rMul() * card];
                cardPartial += (aVal - bVal) * (aVal - bVal);
            }
            final = std::max(final, cardPartial);
        }
        final = T(std::sqrt(final));
        return final;
    }

    T rL2Norm(std::shared_ptr<T[]>& a, std::shared_ptr<T[]>& b)
    {
        T final = 0;
#ifdef _MSC_VER
#else
#pragma omp parallel for collapse(1) reduction(+ \
                                               : final)
#endif
        for (int64_t i = 0; i < m_size3d.rMul(); i++) {
            T cardPartial = 0;
            for (int64_t card = 0; card < m_cardinality; card++) {
                const auto aVal = a.get()[i + m_size3d.rMul() * card];
                const auto bVal = b.get()[i + m_size3d.rMul() * card];
                cardPartial += (aVal - bVal) * (aVal - bVal);
            }
            final += cardPartial;
        }
        final = T(std::sqrt(final));
        return final;
    }


    T dot(std::shared_ptr<T[]>& a, std::shared_ptr<T[]>& b)
    {
        T final = 0;
#ifdef _MSC_VER
#else
#pragma omp parallel for collapse(1) reduction(+ \
                                               : final)
#endif
        for (int64_t i = 0; i < m_size3d.rMul(); i++) {
            T cardPartial = 0;
            for (int64_t card = 0; card < m_cardinality; card++) {
                const auto aVal = a.get()[i + m_size3d.rMul() * card];
                const auto bVal = b.get()[i + m_size3d.rMul() * card];
                cardPartial += (aVal * bVal);
            }
            final += cardPartial;
        }
        return final;
    }

    void sum(T sa, std::shared_ptr<T[]>& a, T sb, std::shared_ptr<T[]>& b, std::shared_ptr<T[]>& c)
    {
#ifdef _MSC_VER
#else
#pragma omp parallel for collapse(2)
#endif
        for (int64_t card = 0; card < m_cardinality; card++) {
            for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                c.get()[i + m_size3d.rMul() * card] = sa * a.get()[i + m_size3d.rMul() * card] + sb * b.get()[i + m_size3d.rMul() * card];
            }
        }
    }

    void scale(std::shared_ptr<T[]>& a, T val, std::shared_ptr<T[]>& c)
    {
#ifdef _MSC_VER
#else
#pragma omp parallel for collapse(2)
#endif
        for (int64_t card = 0; card < m_cardinality; card++) {
            for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                c.get()[i + m_size3d.rMul() * card] = val * a.get()[i + m_size3d.rMul() * card];
            }
        }
    }

    void increment(std::shared_ptr<T[]>& a, std::shared_ptr<T[]>& c)
    {
#pragma omp parallel for collapse(2)

        for (int64_t card = 0; card < m_cardinality; card++) {
            for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                c.get()[i + m_size3d.rMul() * card] += a.get()[i + m_size3d.rMul() * card];
            }
        }
    }

    void decrement(std::shared_ptr<T[]>& a, std::shared_ptr<T[]>& c)
    {
#ifdef _MSC_VER
#else
#pragma omp parallel for collapse(2)
#endif
        for (int64_t card = 0; card < m_cardinality; card++) {
            for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                c.get()[i + m_size3d.rMul() * card] -= a.get()[i + m_size3d.rMul() * card];
            }
        }
    }

    static void gridInit(Neon::index64_3d dim, storage_t<Neon::aGrid, T>& storage)
    {
        storage.m_size3d = {1, 1, 1};
        storage.m_size3d.x = dim.template rMulTyped<int64_t>();

        auto    lengths = storage.m_backend.devSet().template newDataSet<Neon::aGrid::Count>(storage.m_size3d.x / storage.m_backend.devSet().setCardinality());
        int64_t sumTmp = 0;
        for (int i = 0; i < storage.m_size3d.x % storage.m_backend.devSet().setCardinality(); i++) {
            lengths[i]++;
        }
        for (int i = 0; i < storage.m_backend.devSet().setCardinality(); i++) {
            sumTmp += lengths[i];
        }
        assert(sumTmp == storage.m_size3d.x);
        storage.m_grid = Neon::aGrid(storage.m_backend, lengths);

        storage.Xf = storage.m_grid.template newField<T>({storage.m_backend, Neon::DataUse::HOST_DEVICE}, storage.m_cardinality);
        storage.Yf = storage.m_grid.template newField<T>({storage.m_backend, Neon::DataUse::HOST_DEVICE}, storage.m_cardinality);
        storage.Zf = storage.m_grid.template newField<T>({storage.m_backend, Neon::DataUse::HOST_DEVICE}, storage.m_cardinality);
    }

    static void gridInit(Neon::index64_3d                                    dim,
                         storage_t<Neon::domain::details::eGrid::eGrid, T>& storage)
    {
        storage.m_size3d = dim;

        storage.m_grid = Neon::domain::details::eGrid::eGrid(
            storage.m_backend,
            dim.template newType<int32_t>(),
            [&](const Neon::index_3d&) -> bool {
                return true;
            },
            Neon::domain::Stencil::s7_Laplace_t());

        T    outsideVal = 0;
        auto memoryOption = storage.m_backend.getMemoryOptions(Neon::MemoryLayout::structOfArrays);

        storage.Xf = storage.m_grid.template newField<T>("Xf", storage.m_cardinality, T(0), Neon::DataUse::HOST_DEVICE, memoryOption);
        storage.Yf = storage.m_grid.template newField<T>("Yf", storage.m_cardinality, T(0), Neon::DataUse::HOST_DEVICE, memoryOption);
        storage.Zf = storage.m_grid.template newField<T>("Zf", storage.m_cardinality, T(0), Neon::DataUse::HOST_DEVICE, memoryOption);
    }

    static void gridInit(Neon::index64_3d                   dim,
                         storage_t<Neon::dGrid, T>& storage)
    {
        storage.m_size3d = dim;

        storage.m_grid = Neon::dGrid(
            storage.m_backend,
            dim.template newType<int32_t>(),
            [&](const Neon::index_3d&) -> bool {
                return true;
            },
            Neon::domain::Stencil::s7_Laplace_t());

        T    outsideVal = 0;
        auto memoryOption = storage.m_backend.getMemoryOptions(Neon::MemoryLayout::structOfArrays);

        storage.Xf = storage.m_grid.template newField<T>("Xf", storage.m_cardinality, T(0), Neon::DataUse::HOST_DEVICE, memoryOption);
        storage.Yf = storage.m_grid.template newField<T>("Yf", storage.m_cardinality, T(0), Neon::DataUse::HOST_DEVICE, memoryOption);
        storage.Zf = storage.m_grid.template newField<T>("Zf", storage.m_cardinality, T(0), Neon::DataUse::HOST_DEVICE, memoryOption);
    }


    static void gridInit(Neon::index64_3d                   dim,
                         storage_t<Neon::bGrid, T>& storage)
    {
        storage.m_size3d = dim;

        storage.m_grid = Neon::bGrid(
            storage.m_backend,
            dim.template newType<int32_t>(),
            [&](const Neon::index_3d&) -> bool {
                return true;
            },
            Neon::domain::Stencil::s7_Laplace_t());

        T    outsideVal = 0;
        auto memoryOption = storage.m_backend.getMemoryOptions(Neon::MemoryLayout::structOfArrays);

        storage.Xf = storage.m_grid.template newField<T>("Xf", storage.m_cardinality, T(0), Neon::DataUse::HOST_DEVICE, memoryOption);
        storage.Yf = storage.m_grid.template newField<T>("Yf", storage.m_cardinality, T(0), Neon::DataUse::HOST_DEVICE, memoryOption);
        storage.Zf = storage.m_grid.template newField<T>("Zf", storage.m_cardinality, T(0), Neon::DataUse::HOST_DEVICE, memoryOption);
    }
    
    storage_t(Neon::index64_3d dim, int nGPUs, int cardinality, const Neon::Runtime& backendType)
    {
        m_cardinality = cardinality;
        std::vector<int> gpusIds(nGPUs, 0);
        m_backend = Neon::Backend(gpusIds, backendType);
        m_backend.setAvailableStreamSet(3);

        gridInit(dim, *this);


        getNewDenseField(m_size3d, m_cardinality, Xd, Xnd);
        getNewDenseField(m_size3d, m_cardinality, Yd, Ynd);
        getNewDenseField(m_size3d, m_cardinality, Zd, Znd);
    }
};
