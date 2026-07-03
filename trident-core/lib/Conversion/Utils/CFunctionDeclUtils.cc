//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-core/Conversion/Utils/CFunctionDeclUtils.h"
#include "mlir/IR/Builders.h"

namespace trident::conversion::utils {

namespace detail {
mlir::FailureOr<mlir::LLVM::LLVMFuncOp>
getOrCreateCAPIImpl(mlir::ModuleOp moduleOp, llvm::StringRef symbol,
                    mlir::LLVM::LLVMFunctionType expectedType) {
  mlir::LLVM::LLVMFuncOp existingFunc =
      moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(symbol);
  if (existingFunc) {
    if (existingFunc.getFunctionType() != expectedType) {
      moduleOp.emitError() << "existing llvm.func @" << symbol
                           << " has incompatible signature";
      return mlir::failure();
    }
    return existingFunc;
  }

  mlir::OpBuilder builder(moduleOp.getContext());
  builder.setInsertionPointToStart(moduleOp.getBody());
  return mlir::LLVM::LLVMFuncOp::create(builder, moduleOp.getLoc(), symbol,
                                        expectedType);
}
} // namespace detail

mlir::FailureOr<mlir::LLVM::LLVMFuncOp>
getOrCreateCAPI(mlir::ModuleOp moduleOp, llvm::StringRef symbol,
                mlir::LLVM::LLVMFunctionType expectedType) {
  return detail::getOrCreateCAPIImpl(moduleOp, symbol, expectedType);
}

} // namespace trident::conversion::utils
