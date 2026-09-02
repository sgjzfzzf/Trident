//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/DLPack/IR/DLPackDialect.h"
#include "trident/core/Dialect/DLPack/IR/DLPackOps.h"
#include "trident/core/Dialect/DLPack/IR/DLPackTypes.h"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/DialectImplementation.h>

#include "trident/core/Dialect/DLPack/IR/DLPackDialect.cpp.inc"

#define GET_OP_CLASSES
#include "trident/core/Dialect/DLPack/IR/DLPack.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "trident/core/Dialect/DLPack/IR/DLPackTypes.cpp.inc"

namespace trident::dlpack {

void DLPackDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "trident/core/Dialect/DLPack/IR/DLPackTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "trident/core/Dialect/DLPack/IR/DLPack.cpp.inc"
      >();
}

} // namespace trident::dlpack
