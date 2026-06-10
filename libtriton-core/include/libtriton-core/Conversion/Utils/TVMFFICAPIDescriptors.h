//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBTRITON_CORE_CONVERSION_UTILS_TVMFFICAPIDESCRIPTORS_H_
#define LIBTRITON_CORE_CONVERSION_UTILS_TVMFFICAPIDESCRIPTORS_H_

#include "dlpack/dlpack.h"
#include "libtriton-core/Conversion/Utils/CFunctionDeclUtils.h"
#include "tvm/ffi/c_api.h"

// TVM FFI C API function descriptors.
namespace libtriton::conversion::utils {

// TVM FFI tensor conversion.
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE(TVMFFITensorFromDLPack)

// TVM FFI error reporting.
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE(TVMFFIErrorSetRaisedFromCStr)

} // namespace libtriton::conversion::utils

#endif // LIBTRITON_CORE_CONVERSION_UTILS_TVMFFICAPIDESCRIPTORS_H_
