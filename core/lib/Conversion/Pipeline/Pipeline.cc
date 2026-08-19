//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/Pipeline/Pipeline.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "trident/core/Conversion/DecomposeTVMFFI/DecomposeTVMFFI.h"
#include "trident/core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "trident/core/Conversion/TorchExtToGPU/TorchExtToGPU.h"
#include "trident/core/Conversion/TorchExtToLLVM/TorchExtToLLVM.h"
#include "trident/core/Conversion/TorchToCf/TorchToCf.h"
#include "trident/core/Conversion/TorchToTVMFFI/TorchToTVMFFI.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.h"

namespace trident::torch {

#define GEN_PASS_DEF_TRIDENTLOWERINGPIPELINE
#include "trident/core/Conversion/Passes.h.inc"

namespace {

class TridentLoweringPipelinePass
    : public impl::TridentLoweringPipelineBase<TridentLoweringPipelinePass> {
  void runOnOperation() final {
    mlir::OpPassManager pm;
<<<<<<< HEAD
    // TorchToTVMFFI owns semantic conversion and region reference counting.
=======
    pm.addPass(trident::torch::createRAAI());
    pm.addPass(trident::torch::createEliminateRefCountPairs());
>>>>>>> 9fa3f8d (	modified:   trident-core/include/trident-core/Dialect/TorchExt/Transforms/Passes.td)
    pm.addPass(trident::torch::createConvertTorchToCf());
    pm.addPass(trident::torch::createConvertTorchToTVMFFI());
    pm.addPass(trident::torchext::createConvertTorchExtToGPU());
    pm.addPass(mlir::createConvertIndexToLLVMPass());
    pm.addPass(trident::torchext::createConvertTorchExtToLLVM());
    pm.addPass(trident::tvm_ffi::createDecomposeTVMFFI());
    pm.addPass(trident::tvm_ffi::createConvertTVMFFIToLLVM());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createGpuToLLVMConversionPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    if (mlir::failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace trident::torch
