//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "libtriton-core/Conversion/Utils/GlobalString.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FormatVariadic.h"

namespace libtriton::conversion::utils {

mlir::Value getOrCreateGlobalString(mlir::OpBuilder &builder,
                                    mlir::Location loc, mlir::ModuleOp moduleOp,
                                    llvm::StringRef name,
                                    llvm::StringRef content) {
  mlir::MLIRContext *context = moduleOp.getContext();
  const mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);
  const mlir::Type i8Ty = mlir::IntegerType::get(context, 8);
  std::string globalSymName =
      llvm::formatv("__libtriton_constant_{0}_{1}", name, content);
  if (!moduleOp.lookupSymbol<mlir::LLVM::GlobalOp>(globalSymName)) {
    llvm::SmallString<16> nullTerminatedContent = content;
    nullTerminatedContent.push_back(0);
    const mlir::LLVM::LLVMArrayType arrayType =
        mlir::LLVM::LLVMArrayType::get(i8Ty, nullTerminatedContent.size());
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());
    mlir::LLVM::GlobalOp::create(
        builder, loc, arrayType, /*isConstant=*/true,
        mlir::LLVM::linkage::Linkage::Internal, globalSymName,
        /*value=*/builder.getStringAttr(nullTerminatedContent));
  }
  return mlir::LLVM::AddressOfOp::create(builder, loc, ptrTy, globalSymName)
      .getResult();
}

} // namespace libtriton::conversion::utils
