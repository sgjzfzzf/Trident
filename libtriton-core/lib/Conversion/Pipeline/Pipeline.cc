//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "libtriton-core/Conversion/Pipeline/Pipeline.h"
#include "libtriton-core/Conversion/TorchToCf/TorchToCf.h"
#include "libtriton-core/Conversion/TorchToLLVM/FuncBackendTypeConversion.h"
#include "libtriton-core/Conversion/TorchToLLVM/TorchToLLVM.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "libtriton-core/Dialect/TorchExt/Transforms/RewriteTorchAsTorchExt.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"

namespace libtriton::torch {

#define GEN_PASS_DEF_TORCHTOLLVMPIPELINE
#include "libtriton-core/Conversion/Passes.h.inc"

namespace {

class TorchToLLVMPipelinePass
    : public impl::TorchToLLVMPipelineBase<TorchToLLVMPipelinePass> {
public:
  void runOnOperation() final {
    mlir::OpPassManager pm(mlir::ModuleOp::getOperationName());
    pm.addPass(libtriton::torch::createFuncBackendTypeConversion());
    pm.addPass(libtriton::torchext::createRewriteTorchAsTorchExt());
    pm.addPass(libtriton::torch::createConvertTorchToLLVM());
    pm.addPass(libtriton::torch::createConvertTorchToCf());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createConvertToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    if (mlir::failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace libtriton::torch
