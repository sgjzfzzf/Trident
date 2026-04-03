#ifndef TVM_FFI_BINDINGS_DIALECT_TVMFFI_IR_TVMFFIOPS_H_
#define TVM_FFI_BINDINGS_DIALECT_TVMFFI_IR_TVMFFIOPS_H_

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackTypes.h"
#include "tvm_ffi_bindings/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "tvm_ffi_bindings/Dialect/TVMFFI/IR/TVMFFI.h.inc"

#endif // TVM_FFI_BINDINGS_DIALECT_TVMFFI_IR_TVMFFIOPS_H_
