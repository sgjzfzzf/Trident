#ifndef LIBTRITON_CORE_RUNTIME_RUNTIME_H_
#define LIBTRITON_CORE_RUNTIME_RUNTIME_H_

#include <cuda_runtime.h>

#include "torch/csrc/inductor/aoti_torch/c/macros.h"
#include "tvm/ffi/c_api.h"

#define LIBTRITON_CORE_RUNTIME_EXPORT                                          \
  extern "C" __attribute__((visibility("default")))

/// DLPack-compatible deleter callback for TVM FFI tensor conversion.
/// Receives a DLManagedTensor*, extracts the AtenTensorHandle from
/// self->manager_ctx, deletes the tensor, and frees the DLManagedTensor.
LIBTRITON_CORE_RUNTIME_EXPORT void
mLibTritonDLManagedTensorDeleter(struct DLManagedTensor *self);

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

/// Pack an AtenTensorHandle into a TVMFFIAny slot as a kTVMFFITensor object.
///
/// On success, the TVMFFIAny owns the tensor object (ref-counted via TVM FFI).
/// \return 0 on success, non-zero on failure.
LIBTRITON_CORE_RUNTIME_EXPORT int32_t
mLibTritonPackTensorToTVMFFIAny(AtenTensorHandle input, TVMFFIAny *ptr);

/// Unpack a TVMFFIAny slot into an AtenTensorHandle.
///
/// The slot must contain a tensor (kTVMFFIDLTensorPtr or kTVMFFITensor).
/// The caller owns the returned AtenTensorHandle and must delete it via
/// aoti_torch_delete_tensor_object when done.
/// \return 0 on success, non-zero on failure.
LIBTRITON_CORE_RUNTIME_EXPORT int32_t mLibTritonUnpackTVMFFIAnyToTensor(
    const TVMFFIAny *ptr, AtenTensorHandle *output);
#endif // LIBTRITON_CORE_RUNTIME_RUNTIME_H_
