#include "libtriton-core/Conversion/TVMFFIToLLVM/TVMFFILLVMDescriptors.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace libtriton::conversion::utils {

mlir::LLVM::LLVMStructType
TVMFFIAnyLLVMDescriptor::getLLVMType(mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMStructType::getLiteral(
      context,
      {mlir::IntegerType::get(context, 32), mlir::IntegerType::get(context, 32),
       mlir::IntegerType::get(context, 64)});
}

mlir::TypedValue<mlir::IntegerType>
TVMFFIAnyLLVMDescriptor::typeIndex(mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
  return this->template get<0>(builder, loc);
}

mlir::TypedValue<mlir::IntegerType>
TVMFFIAnyLLVMDescriptor::zeroPadding(mlir::OpBuilder &builder,
                                     mlir::Location loc) const {
  return this->template get<1>(builder, loc);
}

mlir::TypedValue<mlir::IntegerType>
TVMFFIAnyLLVMDescriptor::payloadBits(mlir::OpBuilder &builder,
                                     mlir::Location loc) const {
  return this->template get<2>(builder, loc);
}

} // namespace libtriton::conversion::utils
