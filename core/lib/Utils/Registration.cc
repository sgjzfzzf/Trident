//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Utils/Registration.h"
#include "trident/core/Conversion/DLPackToLLVM/DLPackToLLVM.h"
#include "trident/core/Conversion/FinalizeTVMFFI/FinalizeTVMFFI.h"
#include "trident/core/Conversion/Pipeline/Pipeline.h"
#include "trident/core/Conversion/TVMFFIToFunc/TVMFFIToFunc.h"
#include "trident/core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "trident/core/Conversion/TorchExtToGPU/TorchExtToGPU.h"
#include "trident/core/Dialect/DLPack/IR/DLPackDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIInterfaces.h"
#include "trident/core/Dialect/TVMFFI/Transforms/OwnershipDeallocation.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/GPUCommon/GPUToLLVM.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/InitAllPasses.h>

void trident::conversion::registerAllDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllDialects(registry);
  registry.insert<trident::dlpack::DLPackDialect,
                  trident::torchext::TorchExtDialect,
                  trident::tvm_ffi::TVMFFIDialect>();
  mlir::registerAllExtensions(registry);
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::gpu::registerConvertGpuToLLVMInterface(registry);
  trident::conversion::registerConvertTVMFFIToLLVMInterface(registry);
  trident::conversion::registerConvertDLPackToLLVMInterface(registry);
  trident::tvm_ffi::registerTVMFFIObjectOwnershipExternalModels(registry);
}

void trident::conversion::registerAllPasses() {
  mlir::registerAllPasses();
  trident::conversion::registerConvertTorchExtToGPUPass();
  trident::conversion::registerConvertTVMFFIToFuncPass();
  trident::conversion::registerConvertTVMFFIToLLVMPass();
  trident::conversion::registerConvertDLPackToLLVMPass();
  trident::conversion::registerFinalizeTVMFFIPass();
  trident::tvm_ffi::registerOwnershipDeallocationPass();
  mlir::registerReconcileUnrealizedCastsPass();
  trident::conversion::registerTridentLoweringPipelinePass();
}
