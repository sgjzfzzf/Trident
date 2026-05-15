#include "libtriton-core/Conversion/DLPackToLLVM/DLPackLLVMDescriptors.h"

#include <cassert>

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace libtriton::conversion::utils {

mlir::TypedValue<mlir::IntegerType>
DLDeviceLLVMDescriptor::deviceType(mlir::ConversionPatternRewriter &rewriter,
                                   mlir::Location loc) const {
  return this->template get<0>(rewriter, loc);
}

mlir::LLVM::LLVMStructType
DLDeviceLLVMDescriptor::getLLVMType(mlir::MLIRContext *context) {
  mlir::LLVM::LLVMStructType type = mlir::LLVM::LLVMStructType::getLiteral(
      context,
      {mlir::IntegerType::get(context, 32),
       mlir::IntegerType::get(context, 32)},
      /*isPacked=*/true);
  return type;
}

mlir::TypedValue<mlir::IntegerType>
DLDeviceLLVMDescriptor::deviceId(mlir::ConversionPatternRewriter &rewriter,
                                 mlir::Location loc) const {
  return this->template get<1>(rewriter, loc);
}

mlir::TypedValue<mlir::IntegerType>
DLDataTypeLLVMDescriptor::code(mlir::ConversionPatternRewriter &rewriter,
                               mlir::Location loc) const {
  return this->template get<0>(rewriter, loc);
}

mlir::LLVM::LLVMStructType
DLDataTypeLLVMDescriptor::getLLVMType(mlir::MLIRContext *context) {
  mlir::LLVM::LLVMStructType type = mlir::LLVM::LLVMStructType::getLiteral(
      context,
      {mlir::IntegerType::get(context, 8), mlir::IntegerType::get(context, 8),
       mlir::IntegerType::get(context, 16)},
      /*isPacked=*/true);
  return type;
}

mlir::TypedValue<mlir::IntegerType>
DLDataTypeLLVMDescriptor::bits(mlir::ConversionPatternRewriter &rewriter,
                               mlir::Location loc) const {
  return this->template get<1>(rewriter, loc);
}

mlir::TypedValue<mlir::IntegerType>
DLDataTypeLLVMDescriptor::lanes(mlir::ConversionPatternRewriter &rewriter,
                                mlir::Location loc) const {
  return this->template get<2>(rewriter, loc);
}

mlir::TypedValue<mlir::LLVM::LLVMPointerType>
DLTensorLLVMDescriptor::data(mlir::ConversionPatternRewriter &rewriter,
                             mlir::Location loc) const {
  return this->template get<0>(rewriter, loc);
}

mlir::LLVM::LLVMStructType
DLTensorLLVMDescriptor::getLLVMType(mlir::MLIRContext *context) {
  mlir::LLVM::LLVMStructType type = mlir::LLVM::LLVMStructType::getLiteral(
      context,
      {mlir::LLVM::LLVMPointerType::get(context),
       DLDeviceLLVMDescriptor::getLLVMType(context),
       mlir::IntegerType::get(context, 32),
       DLDataTypeLLVMDescriptor::getLLVMType(context),
       mlir::LLVM::LLVMPointerType::get(context),
       mlir::LLVM::LLVMPointerType::get(context),
       mlir::IntegerType::get(context, 64)},
      /*isPacked=*/true);
  return type;
}

DLDeviceLLVMDescriptor
DLTensorLLVMDescriptor::device(mlir::ConversionPatternRewriter &rewriter,
                               mlir::Location loc) const {
  return this->template get<1>(rewriter, loc);
}

mlir::TypedValue<mlir::IntegerType>
DLTensorLLVMDescriptor::ndim(mlir::ConversionPatternRewriter &rewriter,
                             mlir::Location loc) const {
  return this->template get<2>(rewriter, loc);
}

DLDataTypeLLVMDescriptor
DLTensorLLVMDescriptor::dtype(mlir::ConversionPatternRewriter &rewriter,
                              mlir::Location loc) const {
  return this->template get<3>(rewriter, loc);
}

mlir::TypedValue<mlir::LLVM::LLVMPointerType>
DLTensorLLVMDescriptor::shape(mlir::ConversionPatternRewriter &rewriter,
                              mlir::Location loc) const {
  return this->template get<4>(rewriter, loc);
}

mlir::TypedValue<mlir::LLVM::LLVMPointerType>
DLTensorLLVMDescriptor::strides(mlir::ConversionPatternRewriter &rewriter,
                                mlir::Location loc) const {
  return this->template get<5>(rewriter, loc);
}

mlir::TypedValue<mlir::IntegerType>
DLTensorLLVMDescriptor::byteOffset(mlir::ConversionPatternRewriter &rewriter,
                                   mlir::Location loc) const {
  return this->template get<6>(rewriter, loc);
}

DLTensorLLVMDescriptor
DLManagedTensorLLVMDescriptor::tensor(mlir::ConversionPatternRewriter &rewriter,
                                      mlir::Location loc) const {
  return this->template get<0>(rewriter, loc);
}

mlir::LLVM::LLVMStructType
DLManagedTensorLLVMDescriptor::getLLVMType(mlir::MLIRContext *context) {
  mlir::LLVM::LLVMStructType type = mlir::LLVM::LLVMStructType::getLiteral(
      context,
      {DLTensorLLVMDescriptor::getLLVMType(context),
       mlir::LLVM::LLVMPointerType::get(context),
       mlir::LLVM::LLVMPointerType::get(context)},
      /*isPacked=*/true);
  return type;
}

mlir::TypedValue<mlir::LLVM::LLVMPointerType>
DLManagedTensorLLVMDescriptor::managerCtx(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) const {
  return this->template get<1>(rewriter, loc);
}

mlir::TypedValue<mlir::LLVM::LLVMPointerType>
DLManagedTensorLLVMDescriptor::deleter(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) const {
  return this->template get<2>(rewriter, loc);
}

} // namespace libtriton::conversion::utils
