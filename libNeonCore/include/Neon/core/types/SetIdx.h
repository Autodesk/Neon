#pragma once
#include <stdint.h>

#include <iostream>


namespace Neon {

/**
 * Creating an abstraction for an index inside a device set.
 * This is justs a simple wrap around an integer index.
 */
struct SetIdx
{
   private:
    int32_t m_idx{-1} /**< Relative index w.r.t an ordered set */;

   public:
    /**
     *
     * @param idx
     */
    SetIdx(int32_t idx);

    /**
     *
     */
    SetIdx() = default;

    /**
     * Returns the relative index
     */
    int32_t idx() const;

    /**
     * Set the relative index
     */
    void idx(int32_t idx);

    /**
     *
     */
    void setToNext();

    /**
     *
     * @return
     */
    [[nodiscard]] SetIdx getNext() const;

    /**
     *
     * @return
     */
    [[nodiscard]] SetIdx getPrevious() const;

    /**
     *
     * @return
     */
    [[nodiscard]] bool validate() const;


    bool     operator<=(int32_t idx) const;
    bool     operator==(int32_t idx) const;
    bool     operator>=(int32_t idx) const;
    bool     operator<(int32_t idx) const;
    bool     operator!=(int32_t idx) const;
    bool     operator>(int32_t idx) const;
    SetIdx& operator++();
    SetIdx   operator++(int);
    operator int32_t () const;

};

std::ostream& operator<<(std::ostream&  os,
                         SetIdx const& m);


}  // End of namespace Neon
