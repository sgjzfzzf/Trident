#include "libtriton-core-c/Registration.h"

#include "libtriton-core/Conversion/DLPackToLLVM/DLPackToLLVM.h"
#include "libtriton-core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "libtriton-core/Conversion/TorchExtToLLVM/TorchExtToLLVM.h"
#include "libtriton-core/Dialect/DLPack/IR/DLPackDialect.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "libtriton-core/Dialect/TVMFFI/Transforms/Passes.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "libtriton-core/Dialect/TorchExt/Transforms/BufferizableOpInterfaceImpl.h"
#include "libtriton-core/Dialect/TorchExt/Transforms/TorchExtNormalizeOperands.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "torch-mlir-c/Registration.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"

void libtritonCoreRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  libtriton::dlpack::registerConvertDLPackToLLVMInterface(registry);
  libtriton::torch_ext::registerBufferizableOpInterfaceExternalModels(registry);
  libtriton::torch_ext::registerConvertTorchExtToLLVMInterface(registry);
  libtriton::tvm_ffi::registerConvertTVMFFIToLLVMInterface(registry);

  registry.insert<libtriton::dlpack::DLPackDialect,
                  libtriton::torch_ext::TorchExtDialect,
                  libtriton::tvm_ffi::TVMFFIDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect,
                  mlir::torch::TorchConversion::TorchConversionDialect>();
  mlir::registerConvertToLLVMDependentDialectLoading(registry);
  unwrap(context)->appendDialectRegistry(registry);
  torchMlirRegisterAllDialects(context);
  unwrap(context)->loadAllAvailableDialects();
}

void libtritonCoreRegisterAllPasses(void) {
  libtriton::dlpack::registerConvertDLPackToLLVMPass();
  libtriton::tvm_ffi::registerEmitTVMFFIInterfacePass();
  libtriton::torch_ext::registerConvertTorchExtToLLVMPass();
  libtriton::torch_ext::registerNormalizeTorchExtOperandsPass();
  libtriton::tvm_ffi::registerConvertTVMFFIToLLVMPass();
  torchMlirRegisterAllPasses();
}
