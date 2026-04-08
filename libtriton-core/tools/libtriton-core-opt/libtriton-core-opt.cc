// libtriton-core-opt.cc - Standalone optimizer driver for TVMFFI and DLPack
// dialects.
//
// Registers core LLVM conversion pipeline passes and inserts dialects plus
// convert-to-LLVM interfaces needed by tests and development passes so
// libtriton-core-opt can parse and transform `.mlir` files exercising these
// dialects.

#include "libtriton_core/Conversion/DLPackToLLVM/DLPackToLLVM.h"
#include "libtriton_core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "libtriton_core/Dialect/DLPack/IR/DLPackDialect.h"
#include "libtriton_core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::registerConvertToLLVMPass();
  mlir::registerReconcileUnrealizedCastsPass();
  libtriton::dlpack::registerConvertDLPackToLLVMPass();
  libtriton::tvm_ffi::registerTVMFFIToLLVMPasses();

  mlir::DialectRegistry registry;
  libtriton::dlpack::registerConvertDLPackToLLVMInterface(registry);
  libtriton::tvm_ffi::registerConvertTVMFFIToLLVMInterface(registry);
  registry.insert<libtriton::dlpack::DLPackDialect,
                  libtriton::tvm_ffi::TVMFFIDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect>();
  mlir::registerConvertToLLVMDependentDialectLoading(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "TVM FFI modular optimizer driver\n", registry));
}
