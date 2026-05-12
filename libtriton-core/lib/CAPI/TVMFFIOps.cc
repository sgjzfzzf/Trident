#include "libtriton-core-c/TVMFFIOps.h"

#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIOps.h"
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

bool libtritonCoreOperationIsATVMFFIToOp(MlirOperation operation) {
  return isOpType<libtriton::tvm_ffi::ToOp>(operation);
}

bool libtritonCoreOperationIsATVMFFITensorFromDLPackOp(
    MlirOperation operation) {
  return isOpType<libtriton::tvm_ffi::TensorFromDLPackOp>(operation);
}

MlirValue libtritonCoreTVMFFIToGetInput(MlirOperation operation) {
  return getInputValue<libtriton::tvm_ffi::ToOp>(operation);
}

MlirValue libtritonCoreTVMFFIToGetOutput(MlirOperation operation) {
  return getOutputValue<libtriton::tvm_ffi::ToOp>(operation);
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
