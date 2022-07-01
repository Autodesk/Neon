#include <memory.h>

#include <cassert>

#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"

#pragma once

template <typename GridT, typename T>
class Storage
{
   public:
    using Field = typename GridT::template Field<T>;

    Neon::MemoryLayout   m_memOrder;
    GridT                m_grid;
    Neon::index64_3d     m_size3d;
    int                  m_cardinality;
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
        for (int64_t card = 0; card < m_cardinality; card++) {
            for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                dense.get()[i + m_size3d.rMul() * card] = (100 * offset) * (i + 1) + card;
            }
        }
    }

    void setConsant(std::shared_ptr<T[]>& dense, int constVal)
    {
        for (int64_t card = 0; card < m_cardinality; card++) {
            for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                dense.get()[i + m_size3d.rMul() * card] = constVal + card;
            }
        }
    }

    void setLinearly(int offset, std::shared_ptr<T[]>& dense, int targetCard)
    {
        for (int64_t card = 0; card < m_cardinality; card++) {
            if (card == targetCard) {
                for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                    dense.get()[i + m_size3d.rMul() * card] = T((100 * offset) * (i + 1) + card);
                }
            } else {
                for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                    dense.get()[i + m_size3d.rMul() * card] = 0;
                }
            }
        }
    }

    void setConsant(std::shared_ptr<T[]>& dense, int targetCard, T constVal)
    {
        for (int64_t card = 0; card < m_cardinality; card++) {
            if (card == targetCard) {
                for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                    dense.get()[i + m_size3d.rMul() * card] = constVal + T(card);
                }
            } else if (targetCard == -1) {
                for (int64_t i = 0; i < m_size3d.rMul(); i++) {
                    dense.get()[i + m_size3d.rMul() * card] = constVal;
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
    void initLinearly()
    {
        setLinearly(1, Xd);
        setLinearly(2, Yd);
        setLinearly(3, Zd);

        loadField(Xnd, Xf);
        loadField(Ynd, Yf);
        loadField(Znd, Zf);
    }

    void initConst(int targetCard, T a = 1, T b = 2, T c = 3)
    {
        setConsant(Xd, targetCard, a);
        setConsant(Yd, targetCard, b);
        setConsant(Zd, targetCard, c);

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


    T dot(std::shared_ptr<T[]>& a, std::shared_ptr<T[]>& b)
    {
        T final = 0;
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


    T norm2(std::shared_ptr<T[]>& a)
    {
        T final = 0;
        for (int64_t i = 0; i < m_size3d.rMul(); i++) {
            T cardPartial = 0;
            for (int64_t card = 0; card < m_cardinality; card++) {
                const auto aVal = a.get()[i + m_size3d.rMul() * card];
                cardPartial += (aVal * aVal);
            }
            final += cardPartial;
        }
        return std::sqrt(final);
    }

    Storage(Neon::index64_3d                    dim,
            int                                 nGPUs,
            int                                 cardinality,
            const Neon::Runtime& backendType,
            const Neon::MemoryLayout&           memOrder)
    {
        m_memOrder = memOrder;
        m_cardinality = cardinality;
        m_size3d = dim;

        std::vector<int> gpusIds(nGPUs, 0);
        m_backend = Neon::Backend(gpusIds, backendType);

        m_grid = GridT(
            this->m_backend,
            dim.template newType<int32_t>(),
            [&](const Neon::index_3d&) -> bool {
                return true;
            },
            Neon::domain::Stencil::s7_Laplace_t());

        auto memoryOption = m_backend.getMemoryOptions(m_memOrder);

        Xf = m_grid.template newField<T>("Xf", m_cardinality, T(0), Neon::DataUse::IO_COMPUTE, memoryOption);

        Yf = m_grid.template newField<T>("Yf", m_cardinality, T(0), Neon::DataUse::IO_COMPUTE, memoryOption);

        Zf = m_grid.template newField<T>("Zf", m_cardinality, T(0), Neon::DataUse::IO_COMPUTE, memoryOption);

        getNewDenseField(m_size3d, m_cardinality, Xd, Xnd);
        getNewDenseField(m_size3d, m_cardinality, Yd, Ynd);
        getNewDenseField(m_size3d, m_cardinality, Zd, Znd);
    }
};
