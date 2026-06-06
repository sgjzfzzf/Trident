//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_H_
#define LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

#define GET_OP_CLASSES
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFI.h.inc"

#endif // LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_H_
