//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Utils/Registration.h"
#include "trident/core/Conversion/ArithExtToScf/ArithExtToScf.h"
#include "trident/core/Conversion/Pipeline/Pipeline.h"
#include "trident/core/Conversion/TVMFFIToFunc/TVMFFIToFunc.h"
#include "trident/core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "trident/core/Conversion/TorchExtToGPU/TorchExtToGPU.h"
#include "trident/core/Conversion/TorchToCf/TorchToCf.h"
#include "trident/core/Conversion/TorchToScf/TorchToScf.h"
#include "trident/core/Conversion/TorchToTVMFFI/TorchToTVMFFI.h"
#include "trident/core/Dialect/ArithExt/IR/ArithExtDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIInterfaces.h"
#include "trident/core/Dialect/TVMFFI/Transforms/ApplyObjectOwnership.h"
#include "trident/core/Dialect/TVMFFI/Transforms/DecomposeTVMFFI.h"
#include "trident/core/Dialect/Torch/IR/TorchInterfaces.h"
#include "trident/core/Dialect/Torch/Transforms/GeneralizeAtenOps.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/GPUCommon/GPUToLLVM.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/InitAllPasses.h>
#include <torch-mlir/Dialect/Torch/IR/TorchDialect.h>
#include <torch-mlir/Dialect/Torch/Transforms/Passes.h>
#include <torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h>

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
  trident::conversion::registerConvertTVMFFIToLLVMInterface(registry);
  trident::torch::registerTorchToTVMFFITypeInterfaces(registry);
  trident::tvm_ffi::registerTVMFFIObjectOwnershipExternalModels(registry);
}

void trident::conversion::registerAllPasses() {
  mlir::registerAllPasses();
  trident::conversion::registerConvertArithExtToScfPass();
  trident::conversion::registerConvertTorchExtToGPUPass();
  trident::conversion::registerConvertTorchToCfPass();
  trident::conversion::registerConvertTorchToScfPass();
  trident::conversion::registerConvertTorchToTVMFFIPass();
  trident::conversion::registerConvertTVMFFIToFuncPass();
  trident::conversion::registerConvertTVMFFIToLLVMPass();
  trident::tvm_ffi::registerDecomposeTVMFFIPass();
  trident::tvm_ffi::registerApplyObjectOwnershipPass();
  trident::torch::registerGeneralizeAtenOpsPass();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::torch::registerTorchPasses();
  trident::conversion::registerTridentLoweringPipelinePass();
}
