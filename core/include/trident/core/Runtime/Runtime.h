//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_RUNTIME_RUNTIME_H_
#define TRIDENT_CORE_RUNTIME_RUNTIME_H_

#include "torch/csrc/inductor/aoti_torch/c/macros.h"
#include "tvm/ffi/c_api.h"

#define TRIDENT_CORE_RUNTIME_EXPORT                                            \
  extern "C" __attribute__((visibility("default")))

/// Convert a DLPack dtype (code + bits) to a Torch dtype value.
/// Returns the Torch dtype on success, or -1 if the combination is unknown.
TRIDENT_CORE_RUNTIME_EXPORT int32_t
mTridentTVMFFIToTorchType(uint8_t dtype_code, uint8_t dtype_bits);

/// Convert a DLPack device type to a Torch device type value.
/// Falls back to passthrough for unknown device types.
TRIDENT_CORE_RUNTIME_EXPORT int32_t
mTridentTVMFFIDeviceToTorchDeviceType(int32_t dlDeviceType);
/// Convert a Torch dtype value to DLPack dtype. Falls back to kDLFloat/32.
TRIDENT_CORE_RUNTIME_EXPORT DLDataType
mTridentTorchToTVMFFIDtype(int32_t torch_dtype);

/// Convert a Torch device type value to a DLPack device type.
/// Falls back to kDLCPU.
TRIDENT_CORE_RUNTIME_EXPORT DLDeviceType
mTridentTorchToTVMFFIDevice(int32_t torch_device_type);

/// Pack an AtenTensorHandle into a TVMFFIObjectHandle (kTVMFFITensor object).
///
/// On success, the caller owns the returned handle (ref-counted via TVM FFI).
/// \return 0 on success, non-zero on failure.
TRIDENT_CORE_RUNTIME_EXPORT int32_t mTridentTensorToTVMFFIObject(
    AtenTensorHandle input, TVMFFIObjectHandle *out_handle);

#endif // TRIDENT_CORE_RUNTIME_RUNTIME_H_
