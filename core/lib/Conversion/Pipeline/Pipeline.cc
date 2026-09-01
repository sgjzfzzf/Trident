//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/Pipeline/Pipeline.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Conversion/ArithExtToScf/ArithExtToScf.h"
#include "trident/core/Conversion/TVMFFIToFunc/TVMFFIToFunc.h"
#include "trident/core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "trident/core/Conversion/TorchExtToGPU/TorchExtToGPU.h"
#include "trident/core/Conversion/TorchToCf/TorchToCf.h"
#include "trident/core/Conversion/TorchToScf/TorchToScf.h"
#include "trident/core/Conversion/TorchToTVMFFI/TorchToTVMFFI.h"
#include "trident/core/Dialect/TVMFFI/Transforms/ApplyObjectOwnership.h"
#include "trident/core/Dialect/TVMFFI/Transforms/DecomposeTVMFFI.h"
#include "trident/core/Dialect/Torch/Transforms/GeneralizeAtenOps.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.h" // NOLINT(misc-include-cleaner)
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/Passes.h> // NOLINT(misc-include-cleaner)
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

namespace trident::conversion {

#define GEN_PASS_DEF_TRIDENTLOWERINGPIPELINE
#include "trident/core/Conversion/Passes.h.inc"

class TridentLoweringPipelinePass
    : public impl::TridentLoweringPipelineBase<TridentLoweringPipelinePass> {
  void runOnOperation() final {
    mlir::PassManager pm(&getContext(), mlir::ModuleOp::getOperationName());
    pm.addPass(torch::createGeneralizeAtenOps());
    pm.addPass(createConvertArithExtToScf());
    pm.addPass(createConvertTorchToCf());
    pm.addPass(createConvertTorchToScf());
    pm.addPass(createConvertTorchToTVMFFI());
    pm.addPass(tvm_ffi::createApplyObjectOwnership());
    pm.addPass(tvm_ffi::createDecomposeTVMFFI());
    pm.addPass(createConvertTorchExtToGPU());
    pm.addPass(createConvertTVMFFIToFunc());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(createConvertTVMFFIToLLVM());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(
        mlir::createGpuToLLVMConversionPass()); // NOLINT(misc-include-cleaner)
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    if (mlir::failed(pm.run(getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace trident::conversion
