#ifndef LIBTRITON_CORE_DIALECT_TORCHEXT_IR_TORCHEXTOPS_H_
#define LIBTRITON_CORE_DIALECT_TORCHEXT_IR_TORCHEXTOPS_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

#define GET_OP_CLASSES
#include "libtriton-core/Dialect/TorchExt/IR/TorchExt.h.inc"

#endif // LIBTRITON_CORE_DIALECT_TORCHEXT_IR_TORCHEXTOPS_H_
