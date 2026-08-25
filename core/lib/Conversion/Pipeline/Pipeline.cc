//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/Pipeline/Pipeline.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "trident/core/Conversion/ArithExtToScf/ArithExtToScf.h"
#include "trident/core/Conversion/DecomposeTVMFFI/DecomposeTVMFFI.h"
#include "trident/core/Conversion/GeneralizeAtenOps/GeneralizeAtenOps.h"
#include "trident/core/Conversion/TVMFFIToFunc/TVMFFIToFunc.h"
#include "trident/core/Conversion/TorchExtToGPU/TorchExtToGPU.h"
#include "trident/core/Conversion/TorchExtToTVMFFI/TorchExtToTVMFFI.h"
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
    mlir::PassManager pm(&getContext(), mlir::ModuleOp::getOperationName());
    pm.addPass(trident::torch::createGeneralizeAtenOps());
    pm.addPass(trident::arithext::createConvertArithExtToScf());
    pm.addPass(trident::torch::createConvertTorchToCf());
    pm.addPass(trident::torch::createConvertTorchToTVMFFI());
    pm.addPass(trident::torchext::createConvertTorchExtToTVMFFI());
    pm.addPass(trident::tvm_ffi::createDecomposeTVMFFI());
    pm.addPass(trident::tvm_ffi::createConvertTVMFFIToFunc());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::createConvertToLLVMPass());
    pm.addPass(trident::torchext::createConvertTorchExtToGPU());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    if (mlir::failed(pm.run(getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace trident::torch
