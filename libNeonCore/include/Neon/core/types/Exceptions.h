#pragma once

#if !defined(NEON_WARP_COMPILATION)

#include <exception>
#include <iostream>
#include <sstream>

#include "Neon/core/tools/Logger.h"
#include "Neon/core/types/Macros.h"

/**
 * This file define the generic interface for exception provided by Neon Core.
 * Please have a look to the unit test "neonCore_exceptions" for some example of the use.
 *
 * 1. The base class for exceptions is NeonException_t, which extends exception.
 * 2. The exception provide stream capability to store the user message.
 * 3. NEON_THROW should be used instead of c++ throw for NeonException_t type exception. The macro does some extra steps before calling throw.
 */

#define NEON_THROW(EXCEPTION)                                                     \
    {                                                                             \
        (EXCEPTION).setEnvironmentInfo(__FILE__, __LINE__, NEON_FUNCTION_NAME()); \
        (EXCEPTION).logThrow();                                                   \
        throw(EXCEPTION);                                                         \
    }

#define NEON_THROW_UNSUPPORTED_OPTION(...)                                        \
    {                                                                             \
        Neon::NeonException EXCEPTION;                                            \
        EXCEPTION << "Unsupported option";                                        \
                                                                                  \
        (EXCEPTION).setEnvironmentInfo(__FILE__, __LINE__, NEON_FUNCTION_NAME()); \
        (EXCEPTION).logThrow();                                                   \
        throw(EXCEPTION);                                                         \
    }

#define NEON_THROW_UNSUPPORTED_OPERATION(...)                                     \
    {                                                                             \
        Neon::NeonException EXCEPTION;                                            \
        EXCEPTION << "Unsupported operation";                                     \
                                                                                  \
        (EXCEPTION).setEnvironmentInfo(__FILE__, __LINE__, NEON_FUNCTION_NAME()); \
        (EXCEPTION).logThrow();                                                   \
        throw(EXCEPTION);                                                         \
    }
namespace Neon {


class NeonException : public std::exception
{
   public:
    explicit NeonException(std::string componentName);
    explicit NeonException();
    NeonException(const NeonException& other);

    /**
     * Method used by the macro NEON_THROW to set environment information such as line, file name, function name.
     */
    void setEnvironmentInfo(const char* fileName /**< File name */,
                            int         lineNum /**< Line number */,
                            const char* funName /**< Function name */);

    /**
     * String steam capability to store user message.
     */
    template <typename dataType_ta>
    NeonException& operator<<(const dataType_ta& data /**< User information */)
    {
        msg << data;
        m_what = msg.str();
        return *this;
    }

    NeonException& operator<<(const char* data /**< User information */)
    {
        msg << data;
        m_what = msg.str();
        return *this;
    }

    //const char* what() override;
    [[nodiscard]] const char* what() const throw()

    {
        return m_what.c_str();
    }

    /**
     * Method used by the macro NEON_THROW to log this exception has been thrown
     */
    void logThrow();

   private:
    std::string        componentName;
    std::string        m_what;
    std::ostringstream msg;

    const char* m_fileName{nullptr};
    int         m_lineNum{0};
    const char* m_funName{nullptr};
};

}  // End of namespace Neon

#endif
