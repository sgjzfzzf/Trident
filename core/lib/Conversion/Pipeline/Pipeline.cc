//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/Pipeline/Pipeline.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Conversion/ArithExtToScf/ArithExtToScf.h"
#include "trident/core/Conversion/DecomposeTVMFFI/DecomposeTVMFFI.h"
#include "trident/core/Conversion/GeneralizeAtenOps/GeneralizeAtenOps.h"
#include "trident/core/Conversion/TVMFFIToFunc/TVMFFIToFunc.h"
#include "trident/core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "trident/core/Conversion/TorchExtToGPU/TorchExtToGPU.h"
#include "trident/core/Conversion/TorchToCf/TorchToCf.h"
#include "trident/core/Conversion/TorchToTVMFFI/TorchToTVMFFI.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.h" // NOLINT(misc-include-cleaner)
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>     // NOLINT(misc-include-cleaner)
#include <mlir/Dialect/LLVMIR/LLVMDialect.h> // NOLINT(misc-include-cleaner)
#include <mlir/Dialect/SCF/IR/SCF.h>         // NOLINT(misc-include-cleaner)
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h> // NOLINT(misc-include-cleaner)
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/Passes.h>
#include <torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h> // NOLINT(misc-include-cleaner)

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
    pm.addPass(trident::tvm_ffi::createDecomposeTVMFFI());
    pm.addPass(trident::tvm_ffi::createConvertTVMFFIToFunc());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(trident::tvm_ffi::createConvertTVMFFIToLLVM());
    pm.addPass(trident::torchext::createConvertTorchExtToGPU());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createGpuToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    if (mlir::failed(pm.run(getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace trident::torch
