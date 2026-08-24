//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Utils/Registration.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "trident/core/Conversion/ArithExtToScf/ArithExtToScf.h"
#include "trident/core/Conversion/DecomposeTVMFFI/DecomposeTVMFFI.h"
#include "trident/core/Conversion/GeneralizeAtenOps/GeneralizeAtenOps.h"
#include "trident/core/Conversion/Pipeline/Pipeline.h"
#include "trident/core/Conversion/TVMFFIToFunc/TVMFFIToFunc.h"
#include "trident/core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "trident/core/Conversion/TorchConversionToLLVM/TorchConversionToLLVM.h"
#include "trident/core/Conversion/TorchExtToGPU/TorchExtToGPU.h"
#include "trident/core/Conversion/TorchExtToLLVM/TorchExtToLLVM.h"
#include "trident/core/Conversion/TorchToCf/TorchToCf.h"
#include "trident/core/Conversion/TorchToTVMFFI/TorchToTVMFFI.h"
#include "trident/core/Dialect/ArithExt/IR/ArithExtDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.h"

void trident::conversion::registerAllDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllDialects(registry);
  registry.insert<mlir::torch::Torch::TorchDialect,
                  mlir::torch::TorchConversion::TorchConversionDialect,
                  trident::arithext::ArithExtDialect,
                  trident::torchext::TorchExtDialect,
                  trident::tvm_ffi::TVMFFIDialect>();
  mlir::registerAllExtensions(registry);
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::gpu::registerConvertGpuToLLVMInterface(registry);
  trident::torch::registerConvertTorchConversionToLLVMInterface(registry);
  trident::torchext::registerConvertTorchExtToLLVMInterface(registry);
  trident::tvm_ffi::registerConvertTVMFFIToLLVMInterface(registry);
}

void trident::conversion::registerAllPasses() {
  mlir::registerAllPasses();
  trident::arithext::registerConvertArithExtToScfPass();
  mlir::registerConvertToLLVMPass();
  trident::torch::registerConvertTorchConversionToLLVMPass();
  trident::torchext::registerConvertTorchExtToGPUPass();
  trident::torchext::registerConvertTorchExtToLLVMPass();
  trident::torch::registerConvertTorchToCfPass();
  trident::torch::registerConvertTorchToTVMFFIPass();
  trident::tvm_ffi::registerConvertTVMFFIToFuncPass();
  trident::tvm_ffi::registerConvertTVMFFIToLLVMPass();
  trident::tvm_ffi::registerDecomposeTVMFFIPass();
  trident::torch::registerGeneralizeAtenOpsPass();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::torch::registerTorchPasses();
  trident::torch::registerTridentLoweringPipelinePass();
}
