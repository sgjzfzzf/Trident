#ifndef LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_H_
#define LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_H_

#include "libtriton_core/Dialect/DLPack/IR/DLPackTypes.h"
#include "libtriton_core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "libtriton_core/Dialect/TVMFFI/IR/TVMFFI.h.inc"

#endif // LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_H_
