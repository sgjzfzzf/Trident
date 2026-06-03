#ifndef LIBTRITON_CORE_CONVERSION_DLPACKTOLLVM_DLPACKLLVMDESCRIPTORS_H_
#define LIBTRITON_CORE_CONVERSION_DLPACKTOLLVM_DLPACKLLVMDESCRIPTORS_H_

#include <cassert>
#include <cstdint>

#include "libtriton-core/Conversion/Utils/LLVMDescriptorCRTPBase.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace libtriton::conversion::utils {

class DLDeviceLLVMDescriptor
    : public LLVMDescriptorCRTPBase<
          DLDeviceLLVMDescriptor,
          LLVMStructFieldList<mlir::TypedValue<mlir::IntegerType>,
                              mlir::TypedValue<mlir::IntegerType>>> {
public:
  using FieldList = LLVMStructFieldList<mlir::TypedValue<mlir::IntegerType>,
                                        mlir::TypedValue<mlir::IntegerType>>;
  using Base = LLVMDescriptorCRTPBase<DLDeviceLLVMDescriptor, FieldList>;
  using StructValue = typename Base::StructValue;

  using Base::as;
  using Base::Base;
  using Base::build;
  using Base::from;

  static mlir::LLVM::LLVMStructType getLLVMType(mlir::MLIRContext *context);

  mlir::TypedValue<mlir::IntegerType> deviceType(mlir::OpBuilder &builder,
                                                 mlir::Location loc) const;

  mlir::TypedValue<mlir::IntegerType> deviceId(mlir::OpBuilder &builder,
                                               mlir::Location loc) const;

private:
  friend Base;
};

class DLDataTypeLLVMDescriptor
    : public LLVMDescriptorCRTPBase<
          DLDataTypeLLVMDescriptor,
          LLVMStructFieldList<mlir::TypedValue<mlir::IntegerType>,
                              mlir::TypedValue<mlir::IntegerType>,
                              mlir::TypedValue<mlir::IntegerType>>> {
public:
  using FieldList = LLVMStructFieldList<mlir::TypedValue<mlir::IntegerType>,
                                        mlir::TypedValue<mlir::IntegerType>,
                                        mlir::TypedValue<mlir::IntegerType>>;
  using Base = LLVMDescriptorCRTPBase<DLDataTypeLLVMDescriptor, FieldList>;
  using StructValue = typename Base::StructValue;

  using Base::as;
  using Base::Base;
  using Base::build;
  using Base::from;

  static mlir::LLVM::LLVMStructType getLLVMType(mlir::MLIRContext *context);

  mlir::TypedValue<mlir::IntegerType> code(mlir::OpBuilder &builder,
                                           mlir::Location loc) const;

  mlir::TypedValue<mlir::IntegerType> bits(mlir::OpBuilder &builder,
                                           mlir::Location loc) const;

  mlir::TypedValue<mlir::IntegerType> lanes(mlir::OpBuilder &builder,
                                            mlir::Location loc) const;

private:
  friend Base;
};

class DLTensorLLVMDescriptor
    : public LLVMDescriptorCRTPBase<
          DLTensorLLVMDescriptor,
          LLVMStructFieldList<mlir::TypedValue<mlir::LLVM::LLVMPointerType>,
                              DLDeviceLLVMDescriptor,
                              mlir::TypedValue<mlir::IntegerType>,
                              DLDataTypeLLVMDescriptor,
                              mlir::TypedValue<mlir::LLVM::LLVMPointerType>,
                              mlir::TypedValue<mlir::LLVM::LLVMPointerType>,
                              mlir::TypedValue<mlir::IntegerType>>> {
public:
  using FieldList = LLVMStructFieldList<
      mlir::TypedValue<mlir::LLVM::LLVMPointerType>, DLDeviceLLVMDescriptor,
      mlir::TypedValue<mlir::IntegerType>, DLDataTypeLLVMDescriptor,
      mlir::TypedValue<mlir::LLVM::LLVMPointerType>,
      mlir::TypedValue<mlir::LLVM::LLVMPointerType>,
      mlir::TypedValue<mlir::IntegerType>>;
  using Base = LLVMDescriptorCRTPBase<DLTensorLLVMDescriptor, FieldList>;
  using StructValue = typename Base::StructValue;

  using Base::as;
  using Base::Base;
  using Base::build;
  using Base::from;

  static mlir::LLVM::LLVMStructType getLLVMType(mlir::MLIRContext *context);

  mlir::TypedValue<mlir::LLVM::LLVMPointerType> data(mlir::OpBuilder &builder,
                                                     mlir::Location loc) const;

  DLDeviceLLVMDescriptor device(mlir::OpBuilder &builder,
                                mlir::Location loc) const;

  mlir::TypedValue<mlir::IntegerType> ndim(mlir::OpBuilder &builder,
                                           mlir::Location loc) const;

  DLDataTypeLLVMDescriptor dtype(mlir::OpBuilder &builder,
                                 mlir::Location loc) const;

  mlir::TypedValue<mlir::LLVM::LLVMPointerType> shape(mlir::OpBuilder &builder,
                                                      mlir::Location loc) const;

  mlir::TypedValue<mlir::LLVM::LLVMPointerType>
  strides(mlir::OpBuilder &builder, mlir::Location loc) const;

  mlir::TypedValue<mlir::IntegerType> byteOffset(mlir::OpBuilder &builder,
                                                 mlir::Location loc) const;

private:
  friend Base;
};

class DLManagedTensorLLVMDescriptor
    : public LLVMDescriptorCRTPBase<
          DLManagedTensorLLVMDescriptor,
          LLVMStructFieldList<DLTensorLLVMDescriptor,
                              mlir::TypedValue<mlir::LLVM::LLVMPointerType>,
                              mlir::TypedValue<mlir::LLVM::LLVMPointerType>>> {
public:
  using FieldList =
      LLVMStructFieldList<DLTensorLLVMDescriptor,
                          mlir::TypedValue<mlir::LLVM::LLVMPointerType>,
                          mlir::TypedValue<mlir::LLVM::LLVMPointerType>>;
  using Base = LLVMDescriptorCRTPBase<DLManagedTensorLLVMDescriptor, FieldList>;
  using StructValue = typename Base::StructValue;

  using Base::as;
  using Base::Base;
  using Base::build;
  using Base::from;

  static mlir::LLVM::LLVMStructType getLLVMType(mlir::MLIRContext *context);

  DLTensorLLVMDescriptor tensor(mlir::OpBuilder &builder,
                                mlir::Location loc) const;

private:
  friend Base;
};

} // namespace libtriton::conversion::utils

#endif // LIBTRITON_CORE_CONVERSION_DLPACKTOLLVM_DLPACKLLVMDESCRIPTORS_H_
