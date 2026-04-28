#include <cassert>

#include "dlpack/dlpack.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

extern "C" {
void *_mlir_memref_to_llvm_alloc(size_t size);
void *_mlir_memref_to_llvm_aligned_alloc(size_t alignment, size_t size);
void _mlir_memref_to_llvm_free(void *ptr);
void __libtriton_dlpack_default_managed_tensor_deleter(DLManagedTensor *self);

__attribute__((visibility("default"))) void
__mlir_execution_engine_init(llvm::StringMap<void *> &exportSymbols) {
  auto exportSymbol = [&](llvm::StringRef name, auto ptr) {
    assert(exportSymbols.count(name) == 0 && "symbol already exists");
    exportSymbols[name] = reinterpret_cast<void *>(ptr);
  };

  exportSymbol("_mlir_memref_to_llvm_alloc", &_mlir_memref_to_llvm_alloc);
  exportSymbol("_mlir_memref_to_llvm_aligned_alloc",
               &_mlir_memref_to_llvm_aligned_alloc);
  exportSymbol("_mlir_memref_to_llvm_free", &_mlir_memref_to_llvm_free);
  exportSymbol("__libtriton_dlpack_default_managed_tensor_deleter",
               &__libtriton_dlpack_default_managed_tensor_deleter);
}

__attribute__((visibility("default"))) void __mlir_execution_engine_destroy();
}
