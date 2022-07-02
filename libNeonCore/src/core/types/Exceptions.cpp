#include "Neon/core/types/Exceptions.h"

namespace Neon {
NeonException::NeonException(std::string componentName)
    : componentName(std::move(componentName)) {}

NeonException::NeonException()
    : componentName("") {}

NeonException::NeonException(const NeonException& other)
    : componentName(other.componentName),
      m_what(other.m_what),
      m_fileName(other.m_fileName),
      m_lineNum(other.m_lineNum),
      m_funName(other.m_funName)
{
}

void NeonException::setEnvironmentInfo(const char* fileName, int lineNum, const char* funName)
{
    this->m_fileName = fileName;
    this->m_lineNum = lineNum;
    this->m_funName = funName;
}

void NeonException::logThrow()
{
    ::Neon::globalSpace::LoggerObj.getLogger()->error("Exception thrown at \nLine {} File {} Function {} ", m_lineNum, m_fileName, m_funName);
    if (!componentName.empty() && !msg.str().empty()) {
        ::Neon::globalSpace::LoggerObj.getLogger()->error(componentName + ": " + msg.str());
    } else if (!msg.str().empty()) {
        ::Neon::globalSpace::LoggerObj.getLogger()->error(msg.str());
    } else if (!componentName.empty()) {
        ::Neon::globalSpace::LoggerObj.getLogger()->error(componentName);
    }
}


}  // End of namespace Neon
