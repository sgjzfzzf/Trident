#ifndef LIBTRITON_CORE_CONVERSION_UTILS_LLVMDESCRIPTORCRTPBASE_H_
#define LIBTRITON_CORE_CONVERSION_UTILS_LLVMDESCRIPTORCRTPBASE_H_

#include <utility>

#include "libtriton-core/Conversion/Utils/LLVMStructBuilder.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace libtriton::conversion::utils {

template <typename Derived, typename FieldList> class LLVMDescriptorCRTPBase;

template <typename Derived, typename... FieldTypes>
class LLVMDescriptorCRTPBase<Derived, LLVMStructFieldList<FieldTypes...>> {
public:
  using Fields = LLVMStructFieldList<FieldTypes...>;
  using Builder = LLVMStructBuilder<Fields>;
  using StructValue = typename Builder::StructValue;

  explicit LLVMDescriptorCRTPBase(StructValue value);

  static Derived from(mlir::Value value);

  template <typename... Args>
  static Derived build(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::LLVM::LLVMStructType structType, Args &&...args);

  template <std::size_t Index>
  typename Fields::template FieldType<Index> get(mlir::OpBuilder &builder,
                                                 mlir::Location loc) const;

  template <std::size_t Index, typename T>
  void set(mlir::OpBuilder &builder, mlir::Location loc, T &&fieldValue);

  StructValue as() const;

private:
  StructValue value;
};

template <typename Derived, typename... FieldTypes>
LLVMDescriptorCRTPBase<Derived, LLVMStructFieldList<FieldTypes...>>::
    LLVMDescriptorCRTPBase(StructValue value)
    : value(value) {}

template <typename Derived, typename... FieldTypes>
Derived
LLVMDescriptorCRTPBase<Derived, LLVMStructFieldList<FieldTypes...>>::from(
    mlir::Value value) {
  return Derived(mlir::cast<StructValue>(value));
}

template <typename Derived, typename... FieldTypes>
template <typename... Args>
Derived
LLVMDescriptorCRTPBase<Derived, LLVMStructFieldList<FieldTypes...>>::build(
    mlir::OpBuilder &builder, mlir::Location loc,
    mlir::LLVM::LLVMStructType structType, Args &&...args) {
  return Derived(
      Builder::build(builder, loc, structType, std::forward<Args>(args)...));
}

template <typename Derived, typename... FieldTypes>
template <std::size_t Index>
typename LLVMDescriptorCRTPBase<Derived, LLVMStructFieldList<FieldTypes...>>::
    Fields::template FieldType<Index>
    LLVMDescriptorCRTPBase<Derived, LLVMStructFieldList<FieldTypes...>>::get(
        mlir::OpBuilder &builder, mlir::Location loc) const {
  return Builder::template extract<Index>(builder, loc, value);
}

template <typename Derived, typename... FieldTypes>
template <std::size_t Index, typename T>
void LLVMDescriptorCRTPBase<Derived, LLVMStructFieldList<FieldTypes...>>::set(
    mlir::OpBuilder &builder, mlir::Location loc, T &&fieldValue) {
  value = Builder::template insert<Index>(builder, loc, value,
                                          std::forward<T>(fieldValue));
}

template <typename Derived, typename... FieldTypes>
typename LLVMDescriptorCRTPBase<Derived,
                                LLVMStructFieldList<FieldTypes...>>::StructValue
LLVMDescriptorCRTPBase<Derived, LLVMStructFieldList<FieldTypes...>>::as()
    const {
  return value;
}

} // namespace libtriton::conversion::utils

#endif // LIBTRITON_CORE_CONVERSION_UTILS_LLVMDESCRIPTORCRTPBASE_H_
