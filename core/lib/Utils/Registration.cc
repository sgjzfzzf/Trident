//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Utils/Registration.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "trident/core/Conversion/Pipeline/Pipeline.h"
#include "trident/core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "trident/core/Conversion/TorchConversionToLLVM/TorchConversionToLLVM.h"
#include "trident/core/Conversion/TorchExtToGPU/TorchExtToGPU.h"
#include "trident/core/Conversion/TorchExtToLLVM/TorchExtToLLVM.h"
#include "trident/core/Conversion/TorchToCf/TorchToCf.h"
#include "trident/core/Conversion/TorchToLLVM/FuncBackendTypeConversion.h"
#include "trident/core/Conversion/TorchToLLVM/TorchToLLVM.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "trident/core/Dialect/TorchExt/Transforms/EliminateRefCounter.h"
#include "trident/core/Dialect/TorchExt/Transforms/RAAI.h"

void trident::conversion::registerAllPasses() {
  mlir::registerAllPasses();
  mlir::registerConvertToLLVMPass();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::torch::registerTorchPasses();
  trident::torch::registerEliminateRefCountPairsPass();
  trident::torch::registerRAAIPass();
  trident::torchext::registerConvertTorchExtToGPUPass();
  trident::torchext::registerConvertTorchExtToLLVMPass();
  trident::tvm_ffi::registerConvertTVMFFIToLLVMPass();
  trident::torch::registerConvertTorchToCfPass();
  trident::torch::registerFuncBackendTypeConversionPass();
  trident::torch::registerConvertTorchToLLVMPass();
  trident::torch::registerConvertTorchConversionToLLVMPass();
  trident::torch::registerTridentLoweringPipelinePass();
}

void trident::conversion::registerAllDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllDialects(registry);
  registry
      .insert<trident::torchext::TorchExtDialect,
              trident::tvm_ffi::TVMFFIDialect, mlir::torch::Torch::TorchDialect,
              mlir::torch::TorchConversion::TorchConversionDialect>();
  mlir::registerAllExtensions(registry);
  trident::torchext::registerConvertTorchExtToLLVMInterface(registry);
  trident::tvm_ffi::registerConvertTVMFFIToLLVMInterface(registry);
  trident::torch::registerConvertTorchToLLVMInterface(registry);
  trident::torch::registerConvertTorchConversionToLLVMInterface(registry);
}