// libtriton-core-opt.cc - Standalone optimizer driver for TVMFFI and DLPack
// dialects.
//
// Registers core LLVM conversion pipeline passes and inserts dialects plus
// convert-to-LLVM interfaces needed by tests and development passes so
// libtriton-core-opt can parse and transform `.mlir` files exercising these
// dialects.

#include "libtriton-core/Conversion/DLPackToLLVM/DLPackToLLVM.h"
#include "libtriton-core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "libtriton-core/Conversion/TorchExtToLLVM/TorchExtToLLVM.h"
#include "libtriton-core/Dialect/DLPack/IR/DLPackDialect.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "libtriton-core/Dialect/TVMFFI/Transforms/Passes.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "libtriton-core/Dialect/TorchExt/Transforms/TorchExtNormalizeOperands.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::registerConvertToLLVMPass();
  mlir::registerReconcileUnrealizedCastsPass();
  libtriton::dlpack::registerConvertDLPackToLLVMPass();
  libtriton::tvm_ffi::registerEmitTVMFFIInterfacePass();
  libtriton::torch_ext::registerConvertTorchExtToLLVMPass();
  libtriton::torch_ext::registerNormalizeTorchExtOperandsPass();
  libtriton::tvm_ffi::registerConvertTVMFFIToLLVMPass();

  mlir::DialectRegistry registry;
  libtriton::dlpack::registerConvertDLPackToLLVMInterface(registry);
  libtriton::torch_ext::registerConvertTorchExtToLLVMInterface(registry);
  libtriton::tvm_ffi::registerConvertTVMFFIToLLVMInterface(registry);
  registry.insert<libtriton::dlpack::DLPackDialect,
                  libtriton::torch_ext::TorchExtDialect,
                  libtriton::tvm_ffi::TVMFFIDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::torch::Torch::TorchDialect,
                  mlir::torch::TorchConversion::TorchConversionDialect>();
  mlir::registerConvertToLLVMDependentDialectLoading(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "TVM FFI modular optimizer driver\n", registry));
}
