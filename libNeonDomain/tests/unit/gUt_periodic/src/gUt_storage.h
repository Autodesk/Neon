#include <memory.h>

#include <cassert>

#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"

#pragma once

template <typename GridT, typename T>
class Storage
{
   public:
    using field_t = typename GridT::template Field<T>;

    Neon::memLayout_et::order_e m_layout;
    GridT                       m_grid;
    Neon::index64_3d            m_size3d;
    int                         m_cardinality;
    field_t                     Xf, Yf, Zf;
    std::shared_ptr<T[]>        Xd, Yd, Zd;
    Neon::set::Backend_t        m_backend;


   private:
    std::shared_ptr<T[]> getNewDenseField(Neon::index64_3d size3d, int cardinality)
    {
        size_t               size = size3d.template rMulTyped<size_t>();
        std::shared_ptr<T[]> d(new T[size * cardinality],
                               [](T* i) { delete[] i; });
        return d;
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

    void loadField(const std::shared_ptr<T[]>& dense, field_t& field)
    {

        field.ioFromDense(m_layout, dense.get(), T(0));
        field.updateCompute(field.grid().backend().devSet().defaultStreamSet());
        field.grid().backend().devSet().defaultStreamSet().sync();
    }


   public:
    void initLinearly()
    {
        setLinearly(1, Xd);
        setLinearly(2, Yd);
        setLinearly(3, Zd);

        loadField(Xd, Xf);
        loadField(Yd, Yf);
        loadField(Zd, Zf);
    }

    void initConst(int targetCard, T a = 1, T b = 2, T c = 3)
    {
        setConsant(Xd, targetCard, a);
        setConsant(Yd, targetCard, b);
        setConsant(Zd, targetCard, c);

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

    Storage(Neon::index64_3d                           dim,
            int                                        nGPUs,
            int                                        cardinality,
            const Neon::set::Backend_t::runtime_et::e& backendType,
            const Neon::memLayout_et::order_e&         layout)
    {
        m_layout = layout;
        m_cardinality = cardinality;
        m_size3d = dim;

        std::vector<int> gpusIds(nGPUs, 0);
        m_backend = Neon::set::Backend_t(gpusIds, backendType);

        m_grid = GridT(
            this->m_backend,
            dim.template newType<int32_t>(),
            [&](const Neon::index_3d&) -> bool {
                return true;
            },
            Neon::domain::stencil_t::s7_Laplace_t());

        Neon::set::DataConfig_t dataConfig(m_backend,
                                           Neon::DataUse::HOST_DEVICE,
                                           {Neon::DeviceType::CPU, Neon::Allocator::MALLOC, layout},
                                           {Neon::DeviceType::CUDA, Neon::Allocator::CUDA_MEM_DEVICE, layout});

        Xf = m_grid.template newField<T>(dataConfig,
                                         Neon::domain::haloStatus_et::ON,
                                         m_cardinality,
                                         T(0));
        Yf = m_grid.template newField<T>(dataConfig,
                                         Neon::domain::haloStatus_et::ON,
                                         m_cardinality,
                                         T(0));
        Zf = m_grid.template newField<T>(dataConfig,
                                         Neon::domain::haloStatus_et::ON,
                                         m_cardinality,
                                         T(0));

        Xd = getNewDenseField(m_size3d, m_cardinality);
        Yd = getNewDenseField(m_size3d, m_cardinality);
        Zd = getNewDenseField(m_size3d, m_cardinality);
    }
};
