#ifndef LIBTRITON_CORE_CONVERSION_UTILS_LLVMSTRUCTTYPELIST_H_
#define LIBTRITON_CORE_CONVERSION_UTILS_LLVMSTRUCTTYPELIST_H_

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace libtriton::conversion::utils {

template <typename... FieldTypes> struct LLVMStructFieldList {
  static constexpr std::size_t kSize = sizeof...(FieldTypes);

  template <std::size_t Index>
  using FieldType =
      typename std::tuple_element<Index, std::tuple<FieldTypes...>>::type;
};

template <typename T, typename = void> struct HasFrom : std::false_type {};

template <typename T, typename = void> struct HasAs : std::false_type {};

template <typename T>
struct HasAs<T, std::void_t<decltype(std::declval<const std::remove_cv_t<
                                         std::remove_reference_t<T>> &>()
                                         .as())>> : std::true_type {};

template <typename T>
struct HasFrom<
    T, std::void_t<decltype(std::remove_cv_t<std::remove_reference_t<T>>::from(
           std::declval<mlir::Value>()))>> : std::true_type {};

template <typename FieldList> class LLVMStructBuilder;
template <typename Derived, typename FieldList> class LLVMDescriptorCRTPBase;

template <typename... FieldTypes>
class LLVMStructBuilder<LLVMStructFieldList<FieldTypes...>> {
public:
  using Fields = LLVMStructFieldList<FieldTypes...>;
  using StructValue = mlir::TypedValue<mlir::LLVM::LLVMStructType>;

  template <typename... Args>
  static StructValue
  build(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
        mlir::LLVM::LLVMStructType structType, Args &&...args);

  template <std::size_t Index>
  static typename Fields::template FieldType<Index>
  extract(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
          StructValue structValue);

  template <std::size_t Index, typename T>
  static StructValue insert(mlir::ConversionPatternRewriter &rewriter,
                            mlir::Location loc, StructValue structValue,
                            T &&fieldValue);

private:
  template <typename T, std::enable_if_t<HasAs<T>::value, int> = 0>
  static mlir::Value unwrapFieldValue(T &&fieldValue);

  template <typename T, std::enable_if_t<!HasAs<T>::value, int> = 0>
  static mlir::Value unwrapFieldValue(T &&fieldValue);

  template <typename T> static T convertExtracted(mlir::Value extractedValue);

  template <typename Tuple, std::size_t... Indices>
  static StructValue buildImpl(mlir::ConversionPatternRewriter &rewriter,
                               mlir::Location loc, StructValue structValue,
                               Tuple &&fields, std::index_sequence<Indices...>);
};

template <typename Derived, typename... FieldTypes>
class LLVMDescriptorCRTPBase<Derived, LLVMStructFieldList<FieldTypes...>> {
public:
  using Fields = LLVMStructFieldList<FieldTypes...>;
  using Builder = LLVMStructBuilder<Fields>;
  using StructValue = typename Builder::StructValue;

  explicit LLVMDescriptorCRTPBase(StructValue value);

  static Derived from(mlir::Value value);

  template <typename... Args>
  static Derived build(mlir::ConversionPatternRewriter &rewriter,
                       mlir::Location loc,
                       mlir::LLVM::LLVMStructType structType, Args &&...args);

  template <std::size_t Index>
  typename Fields::template FieldType<Index>
  get(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) const;

  template <std::size_t Index, typename T>
  void set(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
           T &&fieldValue);

  StructValue as() const;

private:
  StructValue value;
};

template <typename... FieldTypes>
template <typename... Args>
typename LLVMStructBuilder<LLVMStructFieldList<FieldTypes...>>::StructValue
LLVMStructBuilder<LLVMStructFieldList<FieldTypes...>>::build(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::LLVM::LLVMStructType structType, Args &&...args) {
  static_assert(sizeof...(Args) == sizeof...(FieldTypes),
                "field count does not match the LLVM struct field list");

  StructValue structValue = mlir::cast<StructValue>(
      mlir::LLVM::PoisonOp::create(rewriter, loc, structType).getResult());
  std::tuple<Args &&...> fields(std::forward<Args>(args)...);
  return buildImpl(rewriter, loc, structValue, fields,
                   std::index_sequence_for<FieldTypes...>{});
}

template <typename... FieldTypes>
template <std::size_t Index>
typename LLVMStructBuilder<
    LLVMStructFieldList<FieldTypes...>>::Fields::template FieldType<Index>
LLVMStructBuilder<LLVMStructFieldList<FieldTypes...>>::extract(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    StructValue structValue) {
  using ResultType = typename Fields::template FieldType<Index>;
  mlir::Value extracted =
      mlir::LLVM::ExtractValueOp::create(
          rewriter, loc, structValue,
          llvm::ArrayRef<int64_t>{static_cast<int64_t>(Index)})
          .getResult();
  return convertExtracted<ResultType>(extracted);
}

template <typename... FieldTypes>
template <std::size_t Index, typename T>
typename LLVMStructBuilder<LLVMStructFieldList<FieldTypes...>>::StructValue
LLVMStructBuilder<LLVMStructFieldList<FieldTypes...>>::insert(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    StructValue structValue, T &&fieldValue) {
  return mlir::cast<StructValue>(
      mlir::LLVM::InsertValueOp::create(
          rewriter, loc, structValue,
          unwrapFieldValue(std::forward<T>(fieldValue)),
          llvm::ArrayRef<int64_t>{static_cast<int64_t>(Index)})
          .getResult());
}

template <typename... FieldTypes>
template <typename T, std::enable_if_t<HasAs<T>::value, int>>
mlir::Value
LLVMStructBuilder<LLVMStructFieldList<FieldTypes...>>::unwrapFieldValue(
    T &&fieldValue) {
  return fieldValue.as();
}

template <typename... FieldTypes>
template <typename T, std::enable_if_t<!HasAs<T>::value, int>>
mlir::Value
LLVMStructBuilder<LLVMStructFieldList<FieldTypes...>>::unwrapFieldValue(
    T &&fieldValue) {
  return mlir::Value(fieldValue);
}

template <typename... FieldTypes>
template <typename T>
T LLVMStructBuilder<LLVMStructFieldList<FieldTypes...>>::convertExtracted(
    mlir::Value extractedValue) {
  using Decayed = std::remove_cv_t<std::remove_reference_t<T>>;
  if constexpr (std::is_same_v<Decayed, mlir::Value>) {
    return extractedValue;
  } else if constexpr (HasFrom<Decayed>::value) {
    return Decayed::from(extractedValue);
  } else if constexpr (std::is_constructible_v<Decayed, mlir::Value>) {
    return Decayed(extractedValue);
  } else {
    return mlir::cast<Decayed>(extractedValue);
  }
}

template <typename... FieldTypes>
template <typename Tuple, std::size_t... Indices>
typename LLVMStructBuilder<LLVMStructFieldList<FieldTypes...>>::StructValue
LLVMStructBuilder<LLVMStructFieldList<FieldTypes...>>::buildImpl(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    StructValue structValue, Tuple &&fields, std::index_sequence<Indices...>) {
  StructValue currentValue = structValue;
  ((currentValue =
        insert<Indices>(rewriter, loc, currentValue,
                        std::get<Indices>(std::forward<Tuple>(fields)))),
   ...);
  return currentValue;
}

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
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::LLVM::LLVMStructType structType, Args &&...args) {
  return Derived(
      Builder::build(rewriter, loc, structType, std::forward<Args>(args)...));
}

template <typename Derived, typename... FieldTypes>
template <std::size_t Index>
typename LLVMDescriptorCRTPBase<Derived, LLVMStructFieldList<FieldTypes...>>::
    Fields::template FieldType<Index>
    LLVMDescriptorCRTPBase<Derived, LLVMStructFieldList<FieldTypes...>>::get(
        mlir::ConversionPatternRewriter &rewriter, mlir::Location loc) const {
  return Builder::template extract<Index>(rewriter, loc, value);
}

template <typename Derived, typename... FieldTypes>
template <std::size_t Index, typename T>
void LLVMDescriptorCRTPBase<Derived, LLVMStructFieldList<FieldTypes...>>::set(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    T &&fieldValue) {
  value = Builder::template insert<Index>(rewriter, loc, value,
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

#endif // LIBTRITON_CORE_CONVERSION_UTILS_LLVMSTRUCTTYPELIST_H_
