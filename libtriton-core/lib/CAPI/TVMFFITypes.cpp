#include "libtriton-core-c/TVMFFITypes.h"

#include "libtriton_core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "llvm/Support/Casting.h"

bool libtritonCoreTypeIsATVMFFIAnyType(MlirType type) {
  return llvm::isa<libtriton::tvm_ffi::AnyType>(unwrap(type));
}

bool libtritonCoreTypeIsATVMFFIObjectHandleType(MlirType type) {
  return llvm::isa<libtriton::tvm_ffi::ObjectHandleType>(unwrap(type));
}

MlirType libtritonCoreTVMFFIAnyTypeGet(MlirContext context) {
  return wrap(libtriton::tvm_ffi::AnyType::get(unwrap(context)));
}

MlirType libtritonCoreTVMFFIObjectHandleTypeGet(MlirContext context) {
  return wrap(libtriton::tvm_ffi::ObjectHandleType::get(unwrap(context)));
}

MlirTypeID libtritonCoreTVMFFIAnyTypeGetTypeID(void) {
  return wrap(libtriton::tvm_ffi::AnyType::getTypeID());
}

MlirTypeID libtritonCoreTVMFFIObjectHandleTypeGetTypeID(void) {
  return wrap(libtriton::tvm_ffi::ObjectHandleType::getTypeID());
}
