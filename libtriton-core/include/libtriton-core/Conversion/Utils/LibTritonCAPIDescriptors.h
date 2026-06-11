//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBTRITON_CORE_CONVERSION_UTILS_LIBTRITONCAPIDESCRIPTORS_H_
#define LIBTRITON_CORE_CONVERSION_UTILS_LIBTRITONCAPIDESCRIPTORS_H_

#include "libtriton-core/Conversion/Utils/CFunctionDeclUtils.h"
#include "libtriton-core/Runtime/Runtime.h"

// LibTriton internal C API function descriptors.
namespace libtriton::conversion::utils {

// Torch ↔ TVM FFI type/device mapping.
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(
    mLibTritonTVMFFIDeviceToTorchDeviceType, TVMFFIDeviceToTorchDeviceType)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(mLibTritonTVMFFIToTorchType,
                                           TVMFFIToTorchType)

// Reverse Torch→DLPack dtype/device mapping.
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(mLibTritonTorchToTVMFFIDtype,
                                           TorchToTVMFFIDtype)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(mLibTritonTorchToTVMFFIDevice,
                                           TorchToTVMFFIDevice)

// Runtime tensor pack/unpack helpers.
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(mLibTritonTensorToTVMFFIObject,
                                           TensorToTVMFFIObject)
LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(mLibTritonTVMFFIObjectToTensor,
                                           TVMFFIObjectToTensor)

} // namespace libtriton::conversion::utils

#endif // LIBTRITON_CORE_CONVERSION_UTILS_LIBTRITONCAPIDESCRIPTORS_H_
