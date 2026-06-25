#include "libtriton-core/Utils/Registration.h"

#include "libtriton-core/Conversion/Pipeline/Pipeline.h"
#include "libtriton-core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "libtriton-core/Conversion/TorchConversionToLLVM/TorchConversionToLLVM.h"
#include "libtriton-core/Conversion/TorchExtToGPU/TorchExtToGPU.h"
#include "libtriton-core/Conversion/TorchExtToLLVM/TorchExtToLLVM.h"
#include "libtriton-core/Conversion/TorchToCf/TorchToCf.h"
#include "libtriton-core/Conversion/TorchToLLVM/FuncBackendTypeConversion.h"
#include "libtriton-core/Conversion/TorchToLLVM/TorchToLLVM.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir-c/Registration.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"

void libtriton::conversion::registerAllPasses() {
  mlir::registerAllPasses();
  mlir::registerConvertToLLVMPass();
  mlir::registerReconcileUnrealizedCastsPass();
  libtriton::torchext::registerConvertTorchExtToGPUPass();
  libtriton::torchext::registerConvertTorchExtToLLVMPass();
  libtriton::tvm_ffi::registerConvertTVMFFIToLLVMPass();
  libtriton::torch::registerConvertTorchToCfPass();
  libtriton::torch::registerFuncBackendTypeConversionPass();
  libtriton::torch::registerConvertTorchToLLVMPass();
  libtriton::torch::registerConvertTorchConversionToLLVMPass();
  libtriton::torch::registerTorchToLLVMPipelinePass();
}

void libtriton::conversion::registerAllDialects(
    mlir::DialectRegistry &registry) {
  mlir::registerAllDialects(registry);
  registry.insert<libtriton::torchext::TorchExtDialect,
                  libtriton::tvm_ffi::TVMFFIDialect,
                  mlir::torch::Torch::TorchDialect,
                  mlir::torch::TorchConversion::TorchConversionDialect>();
  mlir::registerAllExtensions(registry);
  libtriton::torchext::registerConvertTorchExtToGPUInterface(registry);
  libtriton::torchext::registerConvertTorchExtToLLVMInterface(registry);
  libtriton::tvm_ffi::registerConvertTVMFFIToLLVMInterface(registry);
  libtriton::torch::registerConvertTorchConversionToLLVMInterface(registry);
}