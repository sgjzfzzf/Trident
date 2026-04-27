#ifndef LIBTRITON_CORE_DIALECT_TVMFFI_TRANSFORMS_PASSES_H_
#define LIBTRITON_CORE_DIALECT_TVMFFI_TRANSFORMS_PASSES_H_

#include "libtriton-core/Dialect/DLPack/IR/DLPackDialect.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace libtriton::tvm_ffi {

#define GEN_PASS_DECL
#include "libtriton-core/Dialect/TVMFFI/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "libtriton-core/Dialect/TVMFFI/Transforms/Passes.h.inc"

} // namespace libtriton::tvm_ffi

#endif // LIBTRITON_CORE_DIALECT_TVMFFI_TRANSFORMS_PASSES_H_

// Re-includable section: expand pass base class definitions.
#ifdef GEN_PASS_DEF_EMITTVMFFIINTERFACE
namespace libtriton::tvm_ffi {
#include "libtriton-core/Dialect/TVMFFI/Transforms/Passes.h.inc"
} // namespace libtriton::tvm_ffi
#undef GEN_PASS_DEF_EMITTVMFFIINTERFACE
#endif
