#ifndef LIBTRITON_CORE_RUNTIME_RUNTIME_H_
#define LIBTRITON_CORE_RUNTIME_RUNTIME_H_

#include <cstdint>

#include <cuda_runtime.h>

#include "c10/core/Device.h"
#include "dlpack/dlpack.h"

#define LIBTRITON_CORE_RUNTIME_EXPORT                                          \
  extern "C" __attribute__((visibility("default")))

LIBTRITON_CORE_RUNTIME_EXPORT void
__libtriton_dlpack_default_managed_tensor_deleter(DLManagedTensor *self);

LIBTRITON_CORE_RUNTIME_EXPORT DLDevice __libtriton_get_current_device();

LIBTRITON_CORE_RUNTIME_EXPORT cudaStream_t
__libtriton_get_current_stream(c10::DeviceIndex device_index);

LIBTRITON_CORE_RUNTIME_EXPORT void *
__libtriton_tvmffi_env_tensor_alloc(const DLDataType dtype, const int32_t ndim,
                                    int64_t *const shape);

#endif // LIBTRITON_CORE_RUNTIME_RUNTIME_H_
