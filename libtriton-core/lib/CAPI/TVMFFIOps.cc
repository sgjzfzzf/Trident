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

template <typename OpTy> MlirValue getStoreValue(MlirOperation operation) {
  mlir::Operation *op = unwrap(operation);
  if (!op) {
    return getNullValue();
  }
  OpTy typedOp = llvm::dyn_cast<OpTy>(op);
  if (!typedOp) {
    return getNullValue();
  }
  return wrap(typedOp.getValue());
}

template <typename OpTy> MlirValue getStorePtr(MlirOperation operation) {
  mlir::Operation *op = unwrap(operation);
  if (!op) {
    return getNullValue();
  }
  OpTy typedOp = llvm::dyn_cast<OpTy>(op);
  if (!typedOp) {
    return getNullValue();
  }
  return wrap(typedOp.getPtr());
}

} // namespace

bool libtritonCoreOperationIsATVMFFIToOp(MlirOperation operation) {
  return isOpType<libtriton::tvm_ffi::ToOp>(operation);
}

MlirValue libtritonCoreTVMFFIToGetInput(MlirOperation operation) {
  return getInputValue<libtriton::tvm_ffi::ToOp>(operation);
}

MlirValue libtritonCoreTVMFFIToGetOutput(MlirOperation operation) {
  return getOutputValue<libtriton::tvm_ffi::ToOp>(operation);
}

bool libtritonCoreOperationIsATVMFFIStoreOp(MlirOperation operation) {
  return isOpType<libtriton::tvm_ffi::StoreOp>(operation);
}

MlirValue libtritonCoreTVMFFIStoreGetValue(MlirOperation operation) {
  return getStoreValue<libtriton::tvm_ffi::StoreOp>(operation);
}

MlirValue libtritonCoreTVMFFIStoreGetPtr(MlirOperation operation) {
  return getStorePtr<libtriton::tvm_ffi::StoreOp>(operation);
}

bool libtritonCoreOperationIsATVMFFIFunctionCallOp(MlirOperation operation) {
  return isOpType<libtriton::tvm_ffi::FunctionCallOp>(operation);
}

MlirValue libtritonCoreTVMFFIFunctionCallGetFunc(MlirOperation operation) {
  mlir::Operation *op = unwrap(operation);
  if (!op) {
    return getNullValue();
  }
  libtriton::tvm_ffi::FunctionCallOp typedOp =
      llvm::dyn_cast<libtriton::tvm_ffi::FunctionCallOp>(op);
  if (!typedOp) {
    return getNullValue();
  }
  return wrap(typedOp.getFunc());
}

int32_t libtritonCoreTVMFFIFunctionCallGetNumArgs(MlirOperation operation) {
  mlir::Operation *op = unwrap(operation);
  if (!op) {
    return 0;
  }
  libtriton::tvm_ffi::FunctionCallOp typedOp =
      llvm::dyn_cast<libtriton::tvm_ffi::FunctionCallOp>(op);
  if (!typedOp) {
    return 0;
  }
  return typedOp.getArgs().size();
}

MlirValue libtritonCoreTVMFFIFunctionCallGetArg(MlirOperation operation,
                                                int32_t index) {
  mlir::Operation *op = unwrap(operation);
  if (!op) {
    return getNullValue();
  }
  libtriton::tvm_ffi::FunctionCallOp typedOp =
      llvm::dyn_cast<libtriton::tvm_ffi::FunctionCallOp>(op);
  if (!typedOp) {
    return getNullValue();
  }
  if (index < 0 || index >= typedOp.getArgs().size()) {
    return getNullValue();
  }
  return wrap(typedOp.getArgs()[index]);
}

MlirValue libtritonCoreTVMFFIFunctionCallGetResult(MlirOperation operation) {
  mlir::Operation *op = unwrap(operation);
  if (!op) {
    return getNullValue();
  }
  libtriton::tvm_ffi::FunctionCallOp typedOp =
      llvm::dyn_cast<libtriton::tvm_ffi::FunctionCallOp>(op);
  if (!typedOp) {
    return getNullValue();
  }
  return wrap(typedOp.getResult());
}
