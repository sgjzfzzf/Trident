#ifndef LIBTRITON_CORE_RUNTIME_RUNTIME_H_
#define LIBTRITON_CORE_RUNTIME_RUNTIME_H_

#include <cuda_runtime.h>

#include "torch/csrc/inductor/aoti_torch/c/macros.h"
#include "torch/csrc/stable/c/shim.h"
#include "tvm/ffi/c_api.h"

#define LIBTRITON_CORE_RUNTIME_EXPORT                                          \
  extern "C" __attribute__((visibility("default")))

/// Convert a DLPack dtype (code + bits) to a Torch dtype value.
/// Returns the Torch dtype on success, or -1 if the combination is unknown.
LIBTRITON_CORE_RUNTIME_EXPORT int32_t
mLibTritonTVMFFIToTorchType(uint8_t dtype_code, uint8_t dtype_bits);

/// Convert a DLPack device type to a Torch device type value.
/// Falls back to passthrough for unknown device types.
LIBTRITON_CORE_RUNTIME_EXPORT int32_t
mLibTritonTVMFFIDeviceToTorchDeviceType(int32_t dlDeviceType);
/// Convert a Torch dtype value to DLPack dtype. Falls back to kDLFloat/32.
LIBTRITON_CORE_RUNTIME_EXPORT DLDataType
mLibTritonTorchToTVMFFIDtype(int32_t torch_dtype);

/// Convert a Torch device type value to a DLPack device type.
/// Falls back to kDLCPU.
LIBTRITON_CORE_RUNTIME_EXPORT DLDeviceType
mLibTritonTorchToTVMFFIDevice(int32_t torch_device_type);

/// Pack an AtenTensorHandle into a TVMFFIObjectHandle (kTVMFFITensor object).
///
/// On success, the caller owns the returned handle (ref-counted via TVM FFI).
/// \return 0 on success, non-zero on failure.
LIBTRITON_CORE_RUNTIME_EXPORT int32_t mLibTritonTensorToTVMFFIObject(
    AtenTensorHandle input, TVMFFIObjectHandle *out_handle);

/// Unpack a TVMFFIObjectHandle into an AtenTensorHandle.
///
/// The handle must be a kTVMFFITensor object (type_index=70).
/// The caller owns the returned AtenTensorHandle and must delete it via
/// aoti_torch_delete_tensor_object when done.
/// \return 0 on success, non-zero on failure.
LIBTRITON_CORE_RUNTIME_EXPORT int32_t mLibTritonTVMFFIObjectToTensor(
    TVMFFIObjectHandle handle, AtenTensorHandle *output);

#endif // LIBTRITON_CORE_RUNTIME_RUNTIME_H_
