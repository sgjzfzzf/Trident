#include "libtriton-core-c/Registration.h"

#include "libtriton_core/Conversion/DLPackToLLVM/DLPackToLLVM.h"
#include "libtriton_core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "libtriton_core/Dialect/DLPack/IR/DLPackDialect.h"
#include "libtriton_core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "mlir/CAPI/IR.h"
#include "torch-mlir-c/Registration.h"

void libtritonCoreRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  registry.insert<libtriton::dlpack::DLPackDialect,
                  libtriton::tvm_ffi::TVMFFIDialect>();
  unwrap(context)->appendDialectRegistry(registry);
  torchMlirRegisterAllDialects(context);
  unwrap(context)->loadAllAvailableDialects();
}

void libtritonCoreRegisterAllPasses(void) {
  libtriton::dlpack::registerConvertDLPackToLLVMPass();
  libtriton::tvm_ffi::registerTVMFFIToLLVMPasses();
  torchMlirRegisterAllPasses();
}
