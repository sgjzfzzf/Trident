#include <cassert>
#include <cstddef>
#include <cstdlib>

#include "c10/cuda/CUDACachingAllocator.h"
#include "c10/cuda/CUDAFunctions.h"
#include "dlpack/dlpack.h"
#include "libtriton-core/Runtime/Runtime.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

LIBTRITON_CORE_RUNTIME_EXPORT void *_mlir_memref_to_llvm_alloc(size_t size) {
  return c10::cuda::CUDACachingAllocator::raw_alloc(size);
}

LIBTRITON_CORE_RUNTIME_EXPORT void *
_mlir_memref_to_llvm_aligned_alloc([[maybe_unused]] size_t alignment,
                                   size_t size) {
  return _mlir_memref_to_llvm_alloc(size);
}

LIBTRITON_CORE_RUNTIME_EXPORT void _mlir_memref_to_llvm_free(void *ptr) {
  c10::cuda::CUDACachingAllocator::raw_delete(ptr);
}

LIBTRITON_CORE_RUNTIME_EXPORT void
__libtriton_dlpack_default_managed_tensor_deleter(DLManagedTensor *self) {
  if (self != NULL) {
    _mlir_memref_to_llvm_free(self->manager_ctx);
    free(self->dl_tensor.strides);
    free(self->dl_tensor.shape);
    free(self);
  }
}

LIBTRITON_CORE_RUNTIME_EXPORT c10::DeviceIndex
__libtriton_get_current_device() {
  return c10::cuda::current_device();
}

LIBTRITON_CORE_RUNTIME_EXPORT cudaStream_t
__libtriton_get_current_stream(c10::DeviceIndex device_index) {
  return c10::cuda::getCurrentCUDAStream(device_index).stream();
}
