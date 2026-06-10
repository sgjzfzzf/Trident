//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBTRITON_CORE_CONVERSION_UTILS_STDLIBCAPIDESCRIPTORS_H_
#define LIBTRITON_CORE_CONVERSION_UTILS_STDLIBCAPIDESCRIPTORS_H_

#include <cstdlib>

#include "libtriton-core/Conversion/Utils/CFunctionDeclUtils.h"

// Standard C library function descriptors.
namespace libtriton::conversion::utils {

LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(malloc, Malloc)

} // namespace libtriton::conversion::utils

#endif // LIBTRITON_CORE_CONVERSION_UTILS_STDLIBCAPIDESCRIPTORS_H_
