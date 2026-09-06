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

mlir::LogicalResult TensorDataOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location>, mlir::ValueRange,
    mlir::DictionaryAttr, mlir::OpaqueProperties, mlir::RegionRange,
    llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(mlir::LLVM::LLVMPointerType::get(context));
  return mlir::success();
}

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
