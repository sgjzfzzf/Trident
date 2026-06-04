#ifndef LIBTRITON_CORE_DIALECT_TVMFFI_TRANSFORMS_PASSES_H_
#define LIBTRITON_CORE_DIALECT_TVMFFI_TRANSFORMS_PASSES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace libtriton::tvm_ffi {

#define GEN_PASS_DECL_FINALIZETVMFFICALL
#include "libtriton-core/Dialect/TVMFFI/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION_FINALIZETVMFFICALL
#include "libtriton-core/Dialect/TVMFFI/Transforms/Passes.h.inc"

} // namespace libtriton::tvm_ffi

#endif // LIBTRITON_CORE_DIALECT_TVMFFI_TRANSFORMS_PASSES_H_
