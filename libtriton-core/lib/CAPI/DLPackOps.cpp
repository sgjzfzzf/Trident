#include "libtriton-core-c/DLPackOps.h"

#include "libtriton_core/Dialect/DLPack/IR/DLPackOps.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "llvm/Support/Casting.h"

namespace {

MlirValue getNullValue() {
  MlirValue value = {nullptr};
  return value;
}

template <typename OpTy> bool isOpType(MlirOperation operation) {
  mlir::Operation *op = unwrap(operation);
  if (!op) {
    return false;
  }
  return llvm::isa<OpTy>(op);
}

template <typename OpTy> MlirValue getInputValue(MlirOperation operation) {
  mlir::Operation *op = unwrap(operation);
  if (!op) {
    return getNullValue();
  }
  OpTy typedOp = llvm::dyn_cast<OpTy>(op);
  if (!typedOp) {
    return getNullValue();
  }
  return wrap(typedOp.getInput());
}

template <typename OpTy> MlirValue getOutputValue(MlirOperation operation) {
  mlir::Operation *op = unwrap(operation);
  if (!op) {
    return getNullValue();
  }
  OpTy typedOp = llvm::dyn_cast<OpTy>(op);
  if (!typedOp) {
    return getNullValue();
  }
  return wrap(typedOp.getOutput());
}

} // namespace

bool libtritonCoreOperationIsADLPackFromMemRefOwnedOp(MlirOperation operation) {
  return isOpType<libtriton::dlpack::FromMemRefOwnedOp>(operation);
}

bool libtritonCoreOperationIsADLPackFromMemRefBorrowedOp(
    MlirOperation operation) {
  return isOpType<libtriton::dlpack::FromMemRefBorrowedOp>(operation);
}

bool libtritonCoreOperationIsADLPackViewOp(MlirOperation operation) {
  return isOpType<libtriton::dlpack::ViewOp>(operation);
}

bool libtritonCoreOperationIsADLPackToMemRefOp(MlirOperation operation) {
  return isOpType<libtriton::dlpack::ToMemRefOp>(operation);
}

MlirValue libtritonCoreDLPackFromMemRefOwnedGetInput(MlirOperation operation) {
  return getInputValue<libtriton::dlpack::FromMemRefOwnedOp>(operation);
}

MlirValue libtritonCoreDLPackFromMemRefOwnedGetOutput(MlirOperation operation) {
  return getOutputValue<libtriton::dlpack::FromMemRefOwnedOp>(operation);
}

MlirValue
libtritonCoreDLPackFromMemRefBorrowedGetInput(MlirOperation operation) {
  return getInputValue<libtriton::dlpack::FromMemRefBorrowedOp>(operation);
}

MlirValue
libtritonCoreDLPackFromMemRefBorrowedGetOutput(MlirOperation operation) {
  return getOutputValue<libtriton::dlpack::FromMemRefBorrowedOp>(operation);
}

MlirValue libtritonCoreDLPackViewGetInput(MlirOperation operation) {
  return getInputValue<libtriton::dlpack::ViewOp>(operation);
}

MlirValue libtritonCoreDLPackViewGetOutput(MlirOperation operation) {
  return getOutputValue<libtriton::dlpack::ViewOp>(operation);
}

MlirValue libtritonCoreDLPackToMemRefGetInput(MlirOperation operation) {
  return getInputValue<libtriton::dlpack::ToMemRefOp>(operation);
}

MlirValue libtritonCoreDLPackToMemRefGetOutput(MlirOperation operation) {
  return getOutputValue<libtriton::dlpack::ToMemRefOp>(operation);
}
