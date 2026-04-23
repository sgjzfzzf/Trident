#ifndef LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TVMFFILLVMDESCRIPTORS_H_
#define LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TVMFFILLVMDESCRIPTORS_H_

#include "libtriton-core/Conversion/Utils/LLVMDescriptorCRTPBase.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace libtriton::conversion::utils {

class TVMFFIObjectHandleLLVMDescriptor {
public:
  static mlir::LLVM::LLVMPointerType getLLVMType(mlir::MLIRContext *context) {
    return mlir::LLVM::LLVMPointerType::get(context);
  }
};

class TVMFFIAnyLLVMDescriptor
    : public LLVMDescriptorCRTPBase<
          TVMFFIAnyLLVMDescriptor,
          LLVMStructFieldList<mlir::TypedValue<mlir::IntegerType>,
                              mlir::TypedValue<mlir::IntegerType>,
                              mlir::TypedValue<mlir::IntegerType>>> {
public:
  using FieldList = LLVMStructFieldList<mlir::TypedValue<mlir::IntegerType>,
                                        mlir::TypedValue<mlir::IntegerType>,
                                        mlir::TypedValue<mlir::IntegerType>>;
  using Base = LLVMDescriptorCRTPBase<TVMFFIAnyLLVMDescriptor, FieldList>;
  using StructValue = typename Base::StructValue;

  using Base::as;
  using Base::Base;
  using Base::build;
  using Base::from;

  static mlir::LLVM::LLVMStructType getLLVMType(mlir::MLIRContext *context) {
    return mlir::LLVM::LLVMStructType::getLiteral(
        context, {mlir::IntegerType::get(context, 32),
                  mlir::IntegerType::get(context, 32),
                  mlir::IntegerType::get(context, 64)});
  }

  mlir::TypedValue<mlir::IntegerType>
  typeIndex(mlir::ConversionPatternRewriter &rewriter,
            mlir::Location loc) const {
    return this->template get<0>(rewriter, loc);
  }

  mlir::TypedValue<mlir::IntegerType>
  zeroPadding(mlir::ConversionPatternRewriter &rewriter,
              mlir::Location loc) const {
    return this->template get<1>(rewriter, loc);
  }

  mlir::TypedValue<mlir::IntegerType>
  payloadBits(mlir::ConversionPatternRewriter &rewriter,
              mlir::Location loc) const {
    return this->template get<2>(rewriter, loc);
  }

private:
  friend Base;
};

} // namespace libtriton::conversion::utils

#endif // LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TVMFFILLVMDESCRIPTORS_H_
