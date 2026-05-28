#include "libtriton-core/Utils/Registration.h"

#include "libtriton-core/Conversion/Bufferization/Bufferization.h"
#include "libtriton-core/Conversion/DLPackToLLVM/DLPackToLLVM.h"
#include "libtriton-core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "libtriton-core/Conversion/TorchExtToLLVM/TorchExtToLLVM.h"
#include "libtriton-core/Dialect/DLPack/IR/DLPackDialect.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "libtriton-core/Dialect/TorchExt/Transforms/BufferizableOpInterfaceImpl.h"
#include "libtriton-core/Dialect/TorchExt/Transforms/TorchExtAsyncKernelLaunch.h"
#include "libtriton-core/Dialect/TorchExt/Transforms/TorchExtNormalizeOperands.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir-c/Registration.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"

void libtriton::conversion::registerAllPasses() {
  mlir::registerAllPasses();
  mlir::registerConvertToLLVMPass();
  mlir::registerReconcileUnrealizedCastsPass();
  libtriton::conversion::registerLibTritonOneShotBufferizePass();
  libtriton::dlpack::registerConvertDLPackToLLVMPass();
  libtriton::torch_ext::registerConvertTorchExtToLLVMPass();
  libtriton::torch_ext::registerAsyncKernelLaunchPass();
  libtriton::torch_ext::registerNormalizeTorchExtOperandsPass();
  libtriton::tvm_ffi::registerConvertTVMFFIToLLVMPass();
  torchMlirRegisterAllPasses();
}

void libtriton::conversion::registerAllDialects(
    mlir::DialectRegistry &registry) {
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  libtriton::dlpack::registerConvertDLPackToLLVMInterface(registry);
  libtriton::torch_ext::registerConvertTorchExtToLLVMInterface(registry);
  libtriton::tvm_ffi::registerConvertTVMFFIToLLVMInterface(registry);
  libtriton::torch_ext::registerBufferizableOpInterfaceExternalModels(registry);

  registry.insert<
      libtriton::dlpack::DLPackDialect, libtriton::torch_ext::TorchExtDialect,
      libtriton::tvm_ffi::TVMFFIDialect, mlir::arith::ArithDialect,
      mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
      mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect, mlir::NVVM::NVVMDialect,
      mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
      mlir::tensor::TensorDialect, mlir::torch::Torch::TorchDialect,
      mlir::torch::TorchConversion::TorchConversionDialect>();
  mlir::registerConvertToLLVMDependentDialectLoading(registry);
}