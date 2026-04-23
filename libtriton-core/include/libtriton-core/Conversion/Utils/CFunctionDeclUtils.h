#ifndef LIBTRITON_CORE_CONVERSION_UTILS_CFUNCTIONDECLUTILS_H_
#define LIBTRITON_CORE_CONVERSION_UTILS_CFUNCTIONDECLUTILS_H_

#include <type_traits>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace libtriton::conversion::utils {

template <typename T> inline constexpr bool kUnsupportedCType = false;

template <typename CType> struct CTypeToLLVM {
  static mlir::Type get(mlir::MLIRContext *context) {
    using BaseType = std::remove_cv_t<std::remove_reference_t<CType>>;

    if constexpr (std::is_integral_v<BaseType>) {
      return mlir::IntegerType::get(context, sizeof(BaseType) * 8);
    } else if constexpr (std::is_same_v<BaseType, float>) {
      return mlir::Float32Type::get(context);
    } else if constexpr (std::is_same_v<BaseType, double>) {
      return mlir::Float64Type::get(context);
    } else if constexpr (std::is_floating_point_v<BaseType>) {
      static_assert(kUnsupportedCType<BaseType>,
                    "Unsupported floating-point C type in CTypeToLLVM");
    } else if constexpr (std::is_pointer_v<BaseType>) {
      return mlir::LLVM::LLVMPointerType::get(context);
    } else {
      static_assert(kUnsupportedCType<BaseType>,
                    "Unsupported C API type in CTypeToLLVM");
    }
  }
};

template <typename FunctionType> struct CFunctionSignature;

template <typename Ret, typename... Args>
struct CFunctionSignature<Ret(Args...)> {
  using RetType = Ret;
  static mlir::LLVM::LLVMFunctionType getLLVMType(mlir::MLIRContext *context) {
    mlir::Type retTy = mlir::LLVM::LLVMVoidType::get(context);
    if constexpr (!std::is_void_v<Ret>) {
      retTy = CTypeToLLVM<Ret>::get(context);
    }

    llvm::SmallVector<mlir::Type> argTypes = {
        CTypeToLLVM<Args>::get(context)...};
    return mlir::LLVM::LLVMFunctionType::get(retTy, argTypes);
  }
};

template <typename FunctionType> struct CFunctionTraits {
  static_assert(kUnsupportedCType<FunctionType>,
                "CFunctionTraits expects a function pointer type");
};

template <typename Ret, typename... Args>
struct CFunctionTraits<Ret (*)(Args...)> {
  using Signature = CFunctionSignature<Ret(Args...)>;
  static mlir::LLVM::LLVMFunctionType getLLVMType(mlir::MLIRContext *context) {
    return Signature::getLLVMType(context);
  }
};

template <typename FunctionType>
mlir::FailureOr<mlir::LLVM::LLVMFuncOp>
getOrCreateCAPI(mlir::ModuleOp moduleOp, llvm::StringRef symbol) {
  mlir::MLIRContext *context = moduleOp.getContext();
  mlir::LLVM::LLVMFunctionType expectedType =
      CFunctionTraits<FunctionType>::getLLVMType(context);

  mlir::LLVM::LLVMFuncOp existingFunc =
      moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(symbol);
  if (existingFunc) {
    if (existingFunc.getFunctionType() != expectedType) {
      moduleOp.emitError() << "existing llvm.func @" << symbol
                           << " has incompatible signature";
      return mlir::failure();
    }
    return existingFunc;
  }

  mlir::OpBuilder builder(context);
  builder.setInsertionPointToStart(moduleOp.getBody());
  return mlir::LLVM::LLVMFuncOp::create(builder, moduleOp.getLoc(), symbol,
                                        expectedType);
}

template <typename Ret, typename... Args>
struct CFunctionTraits<Ret (*)(Args...) noexcept> {
  using Signature = CFunctionSignature<Ret(Args...)>;
  static mlir::LLVM::LLVMFunctionType getLLVMType(mlir::MLIRContext *context) {
    return Signature::getLLVMType(context);
  }
};

} // namespace libtriton::conversion::utils

#define LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(Func, Name)                 \
  inline mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getOrCreate##Name(            \
      mlir::ModuleOp moduleOp) {                                               \
    return ::libtriton::conversion::utils::getOrCreateCAPI<decltype(&::Func)>( \
        moduleOp, #Func);                                                      \
  }

#define LIBTRITON_DECLARE_CAPI_GET_OR_CREATE(Func)                             \
  LIBTRITON_DECLARE_CAPI_GET_OR_CREATE_NAMED(Func, Func)

#endif // LIBTRITON_CORE_CONVERSION_UTILS_CFUNCTIONDECLUTILS_H_