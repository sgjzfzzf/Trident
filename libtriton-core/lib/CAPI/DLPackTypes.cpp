#include "libtriton-core-c/DLPackTypes.h"

#include "libtriton_core/Dialect/DLPack/IR/DLPackTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "llvm/Support/Casting.h"

bool libtritonCoreTypeIsADLPackDLContextType(MlirType type) {
  return llvm::isa<libtriton::dlpack::DLContextType>(unwrap(type));
}

bool libtritonCoreTypeIsADLPackDLDataTypeType(MlirType type) {
  return llvm::isa<libtriton::dlpack::DLDataTypeType>(unwrap(type));
}

bool libtritonCoreTypeIsADLPackDLTensorType(MlirType type) {
  return llvm::isa<libtriton::dlpack::DLTensorType>(unwrap(type));
}

bool libtritonCoreTypeIsADLPackDLManagedTensorType(MlirType type) {
  return llvm::isa<libtriton::dlpack::DLManagedTensorType>(unwrap(type));
}

MlirType libtritonCoreDLPackDLContextTypeGet(MlirContext context) {
  return wrap(libtriton::dlpack::DLContextType::get(unwrap(context)));
}

MlirType libtritonCoreDLPackDLDataTypeTypeGet(MlirContext context) {
  return wrap(libtriton::dlpack::DLDataTypeType::get(unwrap(context)));
}

MlirType libtritonCoreDLPackDLTensorTypeGet(MlirContext context) {
  return wrap(libtriton::dlpack::DLTensorType::get(unwrap(context)));
}

MlirType libtritonCoreDLPackDLManagedTensorTypeGet(MlirContext context) {
  return wrap(libtriton::dlpack::DLManagedTensorType::get(unwrap(context)));
}

MlirTypeID libtritonCoreDLPackDLContextTypeGetTypeID(void) {
  return wrap(libtriton::dlpack::DLContextType::getTypeID());
}

MlirTypeID libtritonCoreDLPackDLDataTypeTypeGetTypeID(void) {
  return wrap(libtriton::dlpack::DLDataTypeType::getTypeID());
}

MlirTypeID libtritonCoreDLPackDLTensorTypeGetTypeID(void) {
  return wrap(libtriton::dlpack::DLTensorType::getTypeID());
}

MlirTypeID libtritonCoreDLPackDLManagedTensorTypeGetTypeID(void) {
  return wrap(libtriton::dlpack::DLManagedTensorType::getTypeID());
}
