//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_UTILS_TRIDENTCAPIDESCRIPTORS_H_
#define TRIDENT_CORE_CONVERSION_UTILS_TRIDENTCAPIDESCRIPTORS_H_

#include "trident-core/Conversion/Utils/CFunctionDeclUtils.h"
#include "trident-core/Runtime/Runtime.h"

// Trident internal C API function descriptors.
namespace trident::conversion::utils {

// Torch ↔ TVM FFI type/device mapping.
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(mTridentTVMFFIDeviceToTorchDeviceType,
                                         TVMFFIDeviceToTorchDeviceType)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(mTridentTVMFFIToTorchType,
                                         TVMFFIToTorchType)

// Reverse Torch→DLPack dtype/device mapping.
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(mTridentTorchToTVMFFIDtype,
                                         TorchToTVMFFIDtype)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(mTridentTorchToTVMFFIDevice,
                                         TorchToTVMFFIDevice)

// Runtime tensor pack/unpack helpers.
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(mTridentTensorToTVMFFIObject,
                                         TensorToTVMFFIObject)
TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(mTridentTVMFFIObjectToTensor,
                                         TVMFFIObjectToTensor)

} // namespace trident::conversion::utils

#endif // TRIDENT_CORE_CONVERSION_UTILS_TRIDENTCAPIDESCRIPTORS_H_
