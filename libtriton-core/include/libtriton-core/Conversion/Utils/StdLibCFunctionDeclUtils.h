#ifndef LIBTRITON_CORE_CONVERSION_UTILS_STDLIBCFUNCTIONDECLUTILS_H_
#define LIBTRITON_CORE_CONVERSION_UTILS_STDLIBCFUNCTIONDECLUTILS_H_

#include <cstdlib>

#include "libtriton-core/Conversion/Utils/CFunctionDeclUtils.h"

namespace libtriton::conversion::utils {

LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(malloc, Malloc)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(free, Free)

} // namespace libtriton::conversion::utils

#endif // LIBTRITON_CORE_CONVERSION_UTILS_STDLIBCFUNCTIONDECLUTILS_H_
