#ifndef TVM_FFI_BINDINGS_CONVERSION_UTILS_DLPACKLLVMDESCRIPTORS_H_
#define TVM_FFI_BINDINGS_CONVERSION_UTILS_DLPACKLLVMDESCRIPTORS_H_

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "tvm_ffi_bindings/Conversion/Utils/LLVMStructTypeList.h"

namespace libtriton::conversion::utils {

class DLContextLLVMDescriptor
    : public LLVMDescriptorCRTPBase<
          DLContextLLVMDescriptor,
          LLVMStructFieldList<mlir::TypedValue<mlir::IntegerType>,
                              mlir::TypedValue<mlir::IntegerType>>> {
public:
  using FieldList = LLVMStructFieldList<mlir::TypedValue<mlir::IntegerType>,
                                        mlir::TypedValue<mlir::IntegerType>>;
  using Base = LLVMDescriptorCRTPBase<DLContextLLVMDescriptor, FieldList>;
  using StructValue = typename Base::StructValue;

  using Base::as;
  using Base::Base;
  using Base::build;
  using Base::from;

  mlir::TypedValue<mlir::IntegerType>
  deviceType(mlir::ConversionPatternRewriter &rewriter,
             mlir::Location loc) const;

  mlir::TypedValue<mlir::IntegerType>
  deviceId(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) const;

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

  mlir::TypedValue<mlir::IntegerType>
  code(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) const;

  mlir::TypedValue<mlir::IntegerType>
  bits(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) const;

  mlir::TypedValue<mlir::IntegerType>
  lanes(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) const;

private:
  friend Base;
};

class DLTensorLLVMDescriptor
    : public LLVMDescriptorCRTPBase<
          DLTensorLLVMDescriptor,
          LLVMStructFieldList<mlir::TypedValue<mlir::LLVM::LLVMPointerType>,
                              DLContextLLVMDescriptor,
                              mlir::TypedValue<mlir::IntegerType>,
                              DLDataTypeLLVMDescriptor,
                              mlir::TypedValue<mlir::LLVM::LLVMPointerType>,
                              mlir::TypedValue<mlir::LLVM::LLVMPointerType>,
                              mlir::TypedValue<mlir::IntegerType>>> {
public:
  using FieldList = LLVMStructFieldList<
      mlir::TypedValue<mlir::LLVM::LLVMPointerType>, DLContextLLVMDescriptor,
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

  mlir::TypedValue<mlir::LLVM::LLVMPointerType>
  data(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) const;

  DLContextLLVMDescriptor ctx(mlir::ConversionPatternRewriter &rewriter,
                              mlir::Location loc) const;

  mlir::TypedValue<mlir::IntegerType>
  ndim(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) const;

  DLDataTypeLLVMDescriptor dtype(mlir::ConversionPatternRewriter &rewriter,
                                 mlir::Location loc) const;

  mlir::TypedValue<mlir::LLVM::LLVMPointerType>
  shape(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) const;

  mlir::TypedValue<mlir::LLVM::LLVMPointerType>
  strides(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) const;

  mlir::TypedValue<mlir::IntegerType>
  byteOffset(mlir::ConversionPatternRewriter &rewriter,
             mlir::Location loc) const;

private:
  friend Base;
};

inline mlir::TypedValue<mlir::IntegerType>
DLContextLLVMDescriptor::deviceType(mlir::ConversionPatternRewriter &rewriter,
                                    mlir::Location loc) const {
  return this->template get<0>(rewriter, loc);
}

inline mlir::TypedValue<mlir::IntegerType>
DLContextLLVMDescriptor::deviceId(mlir::ConversionPatternRewriter &rewriter,
                                  mlir::Location loc) const {
  return this->template get<1>(rewriter, loc);
}

inline mlir::TypedValue<mlir::IntegerType>
DLDataTypeLLVMDescriptor::code(mlir::ConversionPatternRewriter &rewriter,
                               mlir::Location loc) const {
  return this->template get<0>(rewriter, loc);
}

inline mlir::TypedValue<mlir::IntegerType>
DLDataTypeLLVMDescriptor::bits(mlir::ConversionPatternRewriter &rewriter,
                               mlir::Location loc) const {
  return this->template get<1>(rewriter, loc);
}

inline mlir::TypedValue<mlir::IntegerType>
DLDataTypeLLVMDescriptor::lanes(mlir::ConversionPatternRewriter &rewriter,
                                mlir::Location loc) const {
  return this->template get<2>(rewriter, loc);
}

inline mlir::TypedValue<mlir::LLVM::LLVMPointerType>
DLTensorLLVMDescriptor::data(mlir::ConversionPatternRewriter &rewriter,
                             mlir::Location loc) const {
  return this->template get<0>(rewriter, loc);
}

inline DLContextLLVMDescriptor
DLTensorLLVMDescriptor::ctx(mlir::ConversionPatternRewriter &rewriter,
                            mlir::Location loc) const {
  return this->template get<1>(rewriter, loc);
}

inline mlir::TypedValue<mlir::IntegerType>
DLTensorLLVMDescriptor::ndim(mlir::ConversionPatternRewriter &rewriter,
                             mlir::Location loc) const {
  return this->template get<2>(rewriter, loc);
}

inline DLDataTypeLLVMDescriptor
DLTensorLLVMDescriptor::dtype(mlir::ConversionPatternRewriter &rewriter,
                              mlir::Location loc) const {
  return this->template get<3>(rewriter, loc);
}

inline mlir::TypedValue<mlir::LLVM::LLVMPointerType>
DLTensorLLVMDescriptor::shape(mlir::ConversionPatternRewriter &rewriter,
                              mlir::Location loc) const {
  return this->template get<4>(rewriter, loc);
}

inline mlir::TypedValue<mlir::LLVM::LLVMPointerType>
DLTensorLLVMDescriptor::strides(mlir::ConversionPatternRewriter &rewriter,
                                mlir::Location loc) const {
  return this->template get<5>(rewriter, loc);
}

inline mlir::TypedValue<mlir::IntegerType>
DLTensorLLVMDescriptor::byteOffset(mlir::ConversionPatternRewriter &rewriter,
                                   mlir::Location loc) const {
  return this->template get<6>(rewriter, loc);
}

} // namespace libtriton::conversion::utils

#endif // TVM_FFI_BINDINGS_CONVERSION_UTILS_DLPACKLLVMDESCRIPTORS_H_
