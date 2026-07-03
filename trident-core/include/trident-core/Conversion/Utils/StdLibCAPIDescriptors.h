//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_UTILS_STDLIBCAPIDESCRIPTORS_H_
#define TRIDENT_CORE_CONVERSION_UTILS_STDLIBCAPIDESCRIPTORS_H_

#include "trident-core/Conversion/Utils/CFunctionDeclUtils.h"
#include <cstdlib>

// Standard C library function descriptors.
namespace trident::conversion::utils {

TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(malloc, Malloc)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(free, Free)

} // namespace trident::conversion::utils

#endif // TRIDENT_CORE_CONVERSION_UTILS_STDLIBCAPIDESCRIPTORS_H_
