#include "libtriton-core-c/TVMFFIOps.h"

#include "libtriton_core/Dialect/TVMFFI/IR/TVMFFIOps.h"
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

bool libtritonCoreOperationIsATVMFFIFromIntOp(MlirOperation operation) {
  return isOpType<libtriton::tvm_ffi::FromIntOp>(operation);
}

bool libtritonCoreOperationIsATVMFFIToIntOp(MlirOperation operation) {
  return isOpType<libtriton::tvm_ffi::ToIntOp>(operation);
}

bool libtritonCoreOperationIsATVMFFIFromFloatOp(MlirOperation operation) {
  return isOpType<libtriton::tvm_ffi::FromFloatOp>(operation);
}

bool libtritonCoreOperationIsATVMFFIToFloatOp(MlirOperation operation) {
  return isOpType<libtriton::tvm_ffi::ToFloatOp>(operation);
}

bool libtritonCoreOperationIsATVMFFIFromStrOp(MlirOperation operation) {
  return isOpType<libtriton::tvm_ffi::FromStrOp>(operation);
}

bool libtritonCoreOperationIsATVMFFIToStrOp(MlirOperation operation) {
  return isOpType<libtriton::tvm_ffi::ToStrOp>(operation);
}

bool libtritonCoreOperationIsATVMFFIFromTensorOp(MlirOperation operation) {
  return isOpType<libtriton::tvm_ffi::FromTensorOp>(operation);
}

bool libtritonCoreOperationIsATVMFFIToTensorOp(MlirOperation operation) {
  return isOpType<libtriton::tvm_ffi::ToTensorOp>(operation);
}

bool libtritonCoreOperationIsATVMFFIFromObjectOp(MlirOperation operation) {
  return isOpType<libtriton::tvm_ffi::FromObjectOp>(operation);
}

bool libtritonCoreOperationIsATVMFFIToObjectOp(MlirOperation operation) {
  return isOpType<libtriton::tvm_ffi::ToObjectOp>(operation);
}

bool libtritonCoreOperationIsATVMFFITensorFromDLPackOp(
    MlirOperation operation) {
  return isOpType<libtriton::tvm_ffi::TensorFromDLPackOp>(operation);
}

MlirValue libtritonCoreTVMFFIFromIntGetInput(MlirOperation operation) {
  return getInputValue<libtriton::tvm_ffi::FromIntOp>(operation);
}

MlirValue libtritonCoreTVMFFIFromIntGetOutput(MlirOperation operation) {
  return getOutputValue<libtriton::tvm_ffi::FromIntOp>(operation);
}

MlirValue libtritonCoreTVMFFIToIntGetInput(MlirOperation operation) {
  return getInputValue<libtriton::tvm_ffi::ToIntOp>(operation);
}

MlirValue libtritonCoreTVMFFIToIntGetOutput(MlirOperation operation) {
  return getOutputValue<libtriton::tvm_ffi::ToIntOp>(operation);
}

MlirValue libtritonCoreTVMFFIFromFloatGetInput(MlirOperation operation) {
  return getInputValue<libtriton::tvm_ffi::FromFloatOp>(operation);
}

MlirValue libtritonCoreTVMFFIFromFloatGetOutput(MlirOperation operation) {
  return getOutputValue<libtriton::tvm_ffi::FromFloatOp>(operation);
}

MlirValue libtritonCoreTVMFFIToFloatGetInput(MlirOperation operation) {
  return getInputValue<libtriton::tvm_ffi::ToFloatOp>(operation);
}

MlirValue libtritonCoreTVMFFIToFloatGetOutput(MlirOperation operation) {
  return getOutputValue<libtriton::tvm_ffi::ToFloatOp>(operation);
}

MlirValue libtritonCoreTVMFFIFromStrGetInput(MlirOperation operation) {
  return getInputValue<libtriton::tvm_ffi::FromStrOp>(operation);
}

MlirValue libtritonCoreTVMFFIFromStrGetOutput(MlirOperation operation) {
  return getOutputValue<libtriton::tvm_ffi::FromStrOp>(operation);
}

MlirValue libtritonCoreTVMFFIToStrGetInput(MlirOperation operation) {
  return getInputValue<libtriton::tvm_ffi::ToStrOp>(operation);
}

MlirValue libtritonCoreTVMFFIToStrGetOutput(MlirOperation operation) {
  return getOutputValue<libtriton::tvm_ffi::ToStrOp>(operation);
}

MlirValue libtritonCoreTVMFFIFromTensorGetInput(MlirOperation operation) {
  return getInputValue<libtriton::tvm_ffi::FromTensorOp>(operation);
}

MlirValue libtritonCoreTVMFFIFromTensorGetOutput(MlirOperation operation) {
  return getOutputValue<libtriton::tvm_ffi::FromTensorOp>(operation);
}

MlirValue libtritonCoreTVMFFIToTensorGetInput(MlirOperation operation) {
  return getInputValue<libtriton::tvm_ffi::ToTensorOp>(operation);
}

MlirValue libtritonCoreTVMFFIToTensorGetOutput(MlirOperation operation) {
  return getOutputValue<libtriton::tvm_ffi::ToTensorOp>(operation);
}

MlirValue libtritonCoreTVMFFIFromObjectGetInput(MlirOperation operation) {
  return getInputValue<libtriton::tvm_ffi::FromObjectOp>(operation);
}

MlirValue libtritonCoreTVMFFIFromObjectGetOutput(MlirOperation operation) {
  return getOutputValue<libtriton::tvm_ffi::FromObjectOp>(operation);
}

MlirValue libtritonCoreTVMFFIToObjectGetInput(MlirOperation operation) {
  return getInputValue<libtriton::tvm_ffi::ToObjectOp>(operation);
}

MlirValue libtritonCoreTVMFFIToObjectGetOutput(MlirOperation operation) {
  return getOutputValue<libtriton::tvm_ffi::ToObjectOp>(operation);
}

MlirValue libtritonCoreTVMFFITensorFromDLPackGetFrom(MlirOperation operation) {
  mlir::Operation *op = unwrap(operation);
  if (!op) {
    return getNullValue();
  }
  libtriton::tvm_ffi::TensorFromDLPackOp typedOp =
      llvm::dyn_cast<libtriton::tvm_ffi::TensorFromDLPackOp>(op);
  if (!typedOp) {
    return getNullValue();
  }
  return wrap(typedOp.getFrom());
}

MlirValue libtritonCoreTVMFFITensorFromDLPackGetRequireAlignment(
    MlirOperation operation) {
  mlir::Operation *op = unwrap(operation);
  if (!op) {
    return getNullValue();
  }
  libtriton::tvm_ffi::TensorFromDLPackOp typedOp =
      llvm::dyn_cast<libtriton::tvm_ffi::TensorFromDLPackOp>(op);
  if (!typedOp) {
    return getNullValue();
  }
  return wrap(typedOp.getRequireAlignment());
}

MlirValue libtritonCoreTVMFFITensorFromDLPackGetRequireContiguous(
    MlirOperation operation) {
  mlir::Operation *op = unwrap(operation);
  if (!op) {
    return getNullValue();
  }
  libtriton::tvm_ffi::TensorFromDLPackOp typedOp =
      llvm::dyn_cast<libtriton::tvm_ffi::TensorFromDLPackOp>(op);
  if (!typedOp) {
    return getNullValue();
  }
  return wrap(typedOp.getRequireContiguous());
}

MlirValue
libtritonCoreTVMFFITensorFromDLPackGetOutput(MlirOperation operation) {
  return getOutputValue<libtriton::tvm_ffi::TensorFromDLPackOp>(operation);
}
