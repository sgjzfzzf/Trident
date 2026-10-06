//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/Pipeline/Pipeline.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Conversion/DLPackToLLVM/DLPackToLLVM.h"
#include "trident/core/Conversion/FinalizeTVMFFI/FinalizeTVMFFI.h"
#include "trident/core/Conversion/TVMFFIToFunc/TVMFFIToFunc.h"
#include "trident/core/Conversion/TVMFFIToGPU/TVMFFIToGPU.h"
#include "trident/core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "trident/core/Dialect/TVMFFI/Transforms/OwnershipDeallocation.h"
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

namespace trident::conversion {

#define GEN_PASS_DEF_TRIDENTLOWERINGPIPELINE
#include "trident/core/Conversion/Passes.h.inc"

class TridentLoweringPipelinePass final
    : public impl::TridentLoweringPipelineBase<TridentLoweringPipelinePass> {
  void runOnOperation() final {
    mlir::PassManager pm(&getContext(), mlir::ModuleOp::getOperationName());
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(createConvertTVMFFIToGPU());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(createFinalizeTVMFFI());
    pm.addPass(tvm_ffi::createOwnershipDeallocation());
    pm.addPass(createConvertTVMFFIToFunc());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(createConvertTVMFFIToLLVM());
    pm.addPass(createConvertDLPackToLLVM());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(
        mlir::createGpuToLLVMConversionPass()); // NOLINT(misc-include-cleaner)
    pm.addPass(mlir::createSymbolDCEPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    if (mlir::failed(pm.run(getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace trident::conversion
