//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/DLPack/IR/DLPackDialect.h"
#include "trident/core/Dialect/DLPack/IR/DLPackOps.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/DLPack/IR/DLPackTypes.h" // NOLINT(misc-include-cleaner)
#include <llvm/ADT/TypeSwitch.h>             // NOLINT(misc-include-cleaner)
#include <mlir/Dialect/LLVMIR/LLVMDialect.h> // NOLINT(misc-include-cleaner)
#include <mlir/IR/DialectImplementation.h>   // NOLINT(misc-include-cleaner)

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
