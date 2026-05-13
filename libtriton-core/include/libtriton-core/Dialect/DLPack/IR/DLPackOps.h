#ifndef LIBTRITON_CORE_DIALECT_DLPACK_IR_DLPACKOPS_H_
#define LIBTRITON_CORE_DIALECT_DLPACK_IR_DLPACKOPS_H_

#include "libtriton-core/Dialect/DLPack/IR/DLPackTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "libtriton-core/Dialect/DLPack/IR/DLPack.h.inc"

#endif // LIBTRITON_CORE_DIALECT_DLPACK_IR_DLPACKOPS_H_
