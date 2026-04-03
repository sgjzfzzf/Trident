// tvm-ffi-opt.cc - Standalone optimizer driver for TVMFFI and DLPack dialects.
//
// Registers TVMFFI conversion passes and inserts dialects needed by tests and
// development passes into the registry so tvm-ffi-opt can parse and transform
// `.mlir` files exercising these dialects.

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "tvm_ffi_bindings/Conversion/DLPackToLLVM/DLPackToLLVM.h"
#include "tvm_ffi_bindings/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackDialect.h"
#include "tvm_ffi_bindings/Dialect/TVMFFI/IR/TVMFFIDialect.h"

int main(int argc, char **argv) {
  libtriton::dlpack::registerDLPackToLLVMPasses();
  libtriton::tvm_ffi::registerTVMFFIToLLVMPasses();
  mlir::registerConvertToLLVMPass();

  mlir::DialectRegistry registry;
  registry.insert<libtriton::dlpack::DLPackDialect,
                  libtriton::tvm_ffi::TVMFFIDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect>();
  mlir::registerConvertToLLVMDependentDialectLoading(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "TVM FFI modular optimizer driver\n", registry));
}
