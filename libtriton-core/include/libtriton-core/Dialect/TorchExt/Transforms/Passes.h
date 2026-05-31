#ifndef LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_PASSES_H_
#define LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_PASSES_H_

#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace libtriton::torch_ext {

#define GEN_PASS_DECL
#include "libtriton-core/Dialect/TorchExt/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "libtriton-core/Dialect/TorchExt/Transforms/Passes.h.inc"

} // namespace libtriton::torch_ext

#endif // LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_PASSES_H_
