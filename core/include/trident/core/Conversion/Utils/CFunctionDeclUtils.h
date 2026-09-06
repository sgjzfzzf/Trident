//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_UTILS_CFUNCTIONDECLUTILS_H_
#define TRIDENT_CORE_CONVERSION_UTILS_CFUNCTIONDECLUTILS_H_

#include "dlpack/dlpack.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LogicalResult.h>
#include <type_traits>

namespace trident::conversion::utils {

namespace detail {
mlir::FailureOr<mlir::LLVM::LLVMFuncOp>
getOrCreateCAPIImpl(mlir::ModuleOp moduleOp, llvm::StringRef symbol,
                    mlir::LLVM::LLVMFunctionType expectedType);
} // namespace detail

template <typename T> inline constexpr bool kUnsupportedCType = false;

template <typename CType, typename Enable = void> struct CTypeToLLVMImpl {
  static mlir::Type get(mlir::MLIRContext * /*context*/) {
    static_assert(kUnsupportedCType<CType>,
                  "Unsupported C API type in CTypeToLLVM");
  }
};

template <> struct CTypeToLLVMImpl<DLDataType> {
  static mlir::Type get(mlir::MLIRContext *context) {
    return mlir::LLVM::LLVMStructType::getLiteral(
        context,
        {mlir::IntegerType::get(context, 8), mlir::IntegerType::get(context, 8),
         mlir::IntegerType::get(context, 16)},
        /*isPacked=*/true);
  }
};

template <> struct CTypeToLLVMImpl<DLDevice> {
  static mlir::Type get(mlir::MLIRContext *context) {
    return mlir::LLVM::LLVMStructType::getLiteral(
        context,
        {mlir::IntegerType::get(context, 32),
         mlir::IntegerType::get(context, 32)},
        /*isPacked=*/true);
  }
};

template <typename CType>
struct CTypeToLLVMImpl<CType, std::enable_if_t<std::is_integral_v<CType> ||
                                               std::is_enum_v<CType>>> {
  static mlir::Type get(mlir::MLIRContext *context) {
    return mlir::IntegerType::get(context, sizeof(CType) * 8);
  }
};

template <typename CType>
struct CTypeToLLVMImpl<CType, std::enable_if_t<std::is_same_v<CType, float> ||
                                               std::is_same_v<CType, double>>> {
  static mlir::Type get(mlir::MLIRContext *context) {
    using FloatType = std::conditional_t<std::is_same_v<CType, float>,
                                         mlir::Float32Type, mlir::Float64Type>;
    return FloatType::get(context);
  }
};

template <typename CType>
struct CTypeToLLVMImpl<CType,
                       std::enable_if_t<std::is_floating_point_v<CType> &&
                                        !std::is_same_v<CType, float> &&
                                        !std::is_same_v<CType, double>>> {
  static mlir::Type get(mlir::MLIRContext *) {
    static_assert(kUnsupportedCType<CType>,
                  "Unsupported floating-point C type in CTypeToLLVM");
  }
};

template <typename CType>
struct CTypeToLLVMImpl<CType, std::enable_if_t<std::is_pointer_v<CType>>> {
  static mlir::Type get(mlir::MLIRContext *context) {
    return mlir::LLVM::LLVMPointerType::get(context);
  }
};

template <typename CType>
struct CTypeToLLVM
    : CTypeToLLVMImpl<std::remove_cv_t<std::remove_reference_t<CType>>> {};

template <typename Ret> struct CFunctionReturnType {
  static mlir::Type get(mlir::MLIRContext *context) {
    return CTypeToLLVM<Ret>::get(context);
  }
};

template <> struct CFunctionReturnType<void> {
  static mlir::Type get(mlir::MLIRContext *context) {
    return mlir::LLVM::LLVMVoidType::get(context);
  }
};

template <typename FunctionType> struct CFunctionSignature;

template <typename Ret, typename... Args>
struct CFunctionSignature<Ret(Args...)> {
  using RetType = Ret;
  static mlir::LLVM::LLVMFunctionType getLLVMType(mlir::MLIRContext *context) {
    const mlir::Type retTy = CFunctionReturnType<Ret>::get(context);
    const llvm::SmallVector<mlir::Type> argTypes = {
        CTypeToLLVM<Args>::get(context)...};
    return mlir::LLVM::LLVMFunctionType::get(retTy, argTypes);
  }
};

template <typename FunctionType> struct CFunctionTraits {
  static_assert(kUnsupportedCType<FunctionType>,
                "CFunctionTraits expects a function pointer type");
};

template <typename Ret, typename... Args> struct CFunctionTraitsImpl {
  using Signature = CFunctionSignature<Ret(Args...)>;
  static mlir::LLVM::LLVMFunctionType getLLVMType(mlir::MLIRContext *context) {
    return Signature::getLLVMType(context);
  }
};

template <typename Ret, typename... Args>
struct CFunctionTraits<Ret (*)(Args...)> : CFunctionTraitsImpl<Ret, Args...> {};

template <typename FunctionType>
mlir::FailureOr<mlir::LLVM::LLVMFuncOp>
getOrCreateCAPI(mlir::ModuleOp moduleOp, llvm::StringRef symbol) {
  mlir::MLIRContext *context = moduleOp.getContext();
  const mlir::LLVM::LLVMFunctionType expectedType =
      CFunctionTraits<FunctionType>::getLLVMType(context);

  return detail::getOrCreateCAPIImpl(moduleOp, symbol, expectedType);
}

mlir::FailureOr<mlir::LLVM::LLVMFuncOp>
getOrCreateCAPI(mlir::ModuleOp moduleOp, llvm::StringRef symbol,
                mlir::LLVM::LLVMFunctionType expectedType);

template <typename Ret, typename... Args>
struct CFunctionTraits<Ret (*)(Args...) noexcept>
    : CFunctionTraitsImpl<Ret, Args...> {};

} // namespace trident::conversion::utils

#define TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(Func, Name)                   \
  inline mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getOrCreate##Name(            \
      mlir::ModuleOp moduleOp) {                                               \
    return ::trident::conversion::utils::getOrCreateCAPI<decltype(&::Func)>(   \
        moduleOp, #Func);                                                      \
  }

#define TRIDENT_DECLARE_CAPI_GET_OR_CREATE(Func)                               \
  TRIDENT_DECLARE_CAPI_GET_OR_CREATE_NAMED(Func, Func)

#endif // TRIDENT_CORE_CONVERSION_UTILS_CFUNCTIONDECLUTILS_H_
