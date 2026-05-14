#include "libtriton-core-c/DLPackOps.h"

#include "libtriton-core/Dialect/DLPack/IR/DLPackOps.h"
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

bool libtritonCoreOperationIsADLPackNDimOp(MlirOperation operation) {
  return isOpType<libtriton::dlpack::NDimOp>(operation);
}

bool libtritonCoreOperationIsADLPackShapeOp(MlirOperation operation) {
  return isOpType<libtriton::dlpack::ShapeOp>(operation);
}

bool libtritonCoreOperationIsADLPackStridesOp(MlirOperation operation) {
  return isOpType<libtriton::dlpack::StridesOp>(operation);
}

bool libtritonCoreOperationIsADLPackByteOffsetOp(MlirOperation operation) {
  return isOpType<libtriton::dlpack::ByteOffsetOp>(operation);
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

MlirValue libtritonCoreDLPackNDimGetInput(MlirOperation operation) {
  return getInputValue<libtriton::dlpack::NDimOp>(operation);
}

MlirValue libtritonCoreDLPackNDimGetOutput(MlirOperation operation) {
  return getOutputValue<libtriton::dlpack::NDimOp>(operation);
}

MlirValue libtritonCoreDLPackShapeGetInput(MlirOperation operation) {
  return getInputValue<libtriton::dlpack::ShapeOp>(operation);
}

MlirValue libtritonCoreDLPackShapeGetIndex(MlirOperation operation) {
  mlir::Operation *op = unwrap(operation);
  if (!op) {
    return getNullValue();
  }
  libtriton::dlpack::ShapeOp typedOp =
      llvm::dyn_cast<libtriton::dlpack::ShapeOp>(op);
  if (!typedOp) {
    return getNullValue();
  }
  return wrap(typedOp.getIndex());
}

MlirValue libtritonCoreDLPackShapeGetOutput(MlirOperation operation) {
  return getOutputValue<libtriton::dlpack::ShapeOp>(operation);
}

MlirValue libtritonCoreDLPackStridesGetInput(MlirOperation operation) {
  return getInputValue<libtriton::dlpack::StridesOp>(operation);
}

MlirValue libtritonCoreDLPackStridesGetIndex(MlirOperation operation) {
  mlir::Operation *op = unwrap(operation);
  if (!op) {
    return getNullValue();
  }
  libtriton::dlpack::StridesOp typedOp =
      llvm::dyn_cast<libtriton::dlpack::StridesOp>(op);
  if (!typedOp) {
    return getNullValue();
  }
  return wrap(typedOp.getIndex());
}

MlirValue libtritonCoreDLPackStridesGetOutput(MlirOperation operation) {
  return getOutputValue<libtriton::dlpack::StridesOp>(operation);
}

MlirValue libtritonCoreDLPackByteOffsetGetInput(MlirOperation operation) {
  return getInputValue<libtriton::dlpack::ByteOffsetOp>(operation);
}

MlirValue libtritonCoreDLPackByteOffsetGetOutput(MlirOperation operation) {
  return getOutputValue<libtriton::dlpack::ByteOffsetOp>(operation);
}
