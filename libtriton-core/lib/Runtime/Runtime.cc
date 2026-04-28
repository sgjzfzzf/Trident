#include <cassert>
#include <cstddef>
#include <cstdlib>

#include <cuda_runtime.h>

#include "c10/cuda/CUDACachingAllocator.h"
#include "dlpack/dlpack.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#define LIBTRITON_CORE_RUNTIME_EXPORT                                          \
  extern "C" __attribute__((visibility("default")))

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
