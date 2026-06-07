#ifndef LIBTRITON_CORE_RUNTIME_RUNTIME_H_
#define LIBTRITON_CORE_RUNTIME_RUNTIME_H_

#include <cuda_runtime.h>

#include "torch/csrc/inductor/aoti_torch/c/macros.h"
#include "tvm/ffi/c_api.h"

#define LIBTRITON_CORE_RUNTIME_EXPORT                                          \
  extern "C" __attribute__((visibility("default")))

LIBTRITON_CORE_RUNTIME_EXPORT void
mLibTritonDLManagedTensorDeleter(AtenTensorHandle ctx);

/// Convert a DLPack dtype (code + bits) to a Torch dtype value.
/// Returns the Torch dtype on success, or -1 if the combination is unknown.
LIBTRITON_CORE_RUNTIME_EXPORT int32_t
mLibTritonTVMFFIToTorchType(uint8_t dtype_code, uint8_t dtype_bits);

/// Convert a DLPack device type to a Torch device type value.
/// Falls back to passthrough for unknown device types.
LIBTRITON_CORE_RUNTIME_EXPORT int32_t
mLibTritonTVMFFIDeviceToTorchDeviceType(int32_t dlDeviceType);

#endif // LIBTRITON_CORE_RUNTIME_RUNTIME_H_
