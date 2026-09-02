//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_DLPACK_IR_DLPACKTYPES_H_
#define TRIDENT_CORE_DIALECT_DLPACK_IR_DLPACKTYPES_H_

#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>

#define GET_TYPEDEF_CLASSES
#include "trident/core/Dialect/DLPack/IR/DLPackTypes.h.inc"

#endif // TRIDENT_CORE_DIALECT_DLPACK_IR_DLPACKTYPES_H_
