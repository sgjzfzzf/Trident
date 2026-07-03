//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_UTILS_TVMFFICAPIDESCRIPTORS_H_
#define TRIDENT_CORE_CONVERSION_UTILS_TVMFFICAPIDESCRIPTORS_H_

#include "trident-core/Conversion/Utils/CFunctionDeclUtils.h"
#include "tvm/ffi/c_api.h"

// TVM FFI C API function descriptors.
namespace trident::conversion::utils {

// TVM FFI tensor conversion.
TRIDENT_DECLARE_CAPI_GET_OR_CREATE(TVMFFITensorFromDLPack)

// TVM FFI error reporting.
TRIDENT_DECLARE_CAPI_GET_OR_CREATE(TVMFFIErrorSetRaisedFromCStr)

// TVM FFI function dispatch (used to call ffi.Array, ffi.ArraySize, etc.).
TRIDENT_DECLARE_CAPI_GET_OR_CREATE(TVMFFIFunctionGetGlobal)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE(TVMFFIFunctionCall)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE(TVMFFIObjectIncRef)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE(TVMFFIObjectDecRef)

} // namespace trident::conversion::utils

#endif // TRIDENT_CORE_CONVERSION_UTILS_TVMFFICAPIDESCRIPTORS_H_
