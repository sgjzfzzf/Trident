#include "libtriton-core/Conversion/DLPackToLLVM/DLPackLLVMDescriptors.h"

#include <cassert>

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace libtriton::conversion::utils {

mlir::TypedValue<mlir::IntegerType>
DLDeviceLLVMDescriptor::deviceType(mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
  return this->template get<0>(builder, loc);
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
DLDeviceLLVMDescriptor::deviceId(mlir::OpBuilder &builder,
                                 mlir::Location loc) const {
  return this->template get<1>(builder, loc);
}

mlir::TypedValue<mlir::IntegerType>
DLDataTypeLLVMDescriptor::code(mlir::OpBuilder &builder,
                               mlir::Location loc) const {
  return this->template get<0>(builder, loc);
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
DLDataTypeLLVMDescriptor::bits(mlir::OpBuilder &builder,
                               mlir::Location loc) const {
  return this->template get<1>(builder, loc);
}

mlir::TypedValue<mlir::IntegerType>
DLDataTypeLLVMDescriptor::lanes(mlir::OpBuilder &builder,
                                mlir::Location loc) const {
  return this->template get<2>(builder, loc);
}

mlir::TypedValue<mlir::LLVM::LLVMPointerType>
DLTensorLLVMDescriptor::data(mlir::OpBuilder &builder,
                             mlir::Location loc) const {
  return this->template get<0>(builder, loc);
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
DLTensorLLVMDescriptor::device(mlir::OpBuilder &builder,
                               mlir::Location loc) const {
  return this->template get<1>(builder, loc);
}

mlir::TypedValue<mlir::IntegerType>
DLTensorLLVMDescriptor::ndim(mlir::OpBuilder &builder,
                             mlir::Location loc) const {
  return this->template get<2>(builder, loc);
}

DLDataTypeLLVMDescriptor
DLTensorLLVMDescriptor::dtype(mlir::OpBuilder &builder,
                              mlir::Location loc) const {
  return this->template get<3>(builder, loc);
}

mlir::TypedValue<mlir::LLVM::LLVMPointerType>
DLTensorLLVMDescriptor::shape(mlir::OpBuilder &builder,
                              mlir::Location loc) const {
  return this->template get<4>(builder, loc);
}

mlir::TypedValue<mlir::LLVM::LLVMPointerType>
DLTensorLLVMDescriptor::strides(mlir::OpBuilder &builder,
                                mlir::Location loc) const {
  return this->template get<5>(builder, loc);
}

mlir::TypedValue<mlir::IntegerType>
DLTensorLLVMDescriptor::byteOffset(mlir::OpBuilder &builder,
                                   mlir::Location loc) const {
  return this->template get<6>(builder, loc);
}

DLTensorLLVMDescriptor
DLManagedTensorLLVMDescriptor::tensor(mlir::OpBuilder &builder,
                                      mlir::Location loc) const {
  return this->template get<0>(builder, loc);
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

} // namespace libtriton::conversion::utils
