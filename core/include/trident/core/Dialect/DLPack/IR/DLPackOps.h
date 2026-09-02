//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_DLPACK_IR_DLPACKOPS_H_
#define TRIDENT_CORE_DIALECT_DLPACK_IR_DLPACKOPS_H_

#include "trident/core/Dialect/DLPack/IR/DLPackTypes.h"
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#define GET_OP_CLASSES
#include "trident/core/Dialect/DLPack/IR/DLPack.h.inc"

#endif // TRIDENT_CORE_DIALECT_DLPACK_IR_DLPACKOPS_H_
