//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/DLPack/IR/DLPackTypes.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <optional>
#include <torch-mlir/Dialect/Torch/IR/TorchTypes.h>

namespace trident::tvm_ffi {

mlir::LogicalResult FuncOp::verify() {
  if (std::optional<llvm::StringRef> visibility = getSymVisibility();
      visibility && *visibility != "public") {
    return emitOpError("must have public visibility");
  }
  return mlir::success();
}

mlir::LogicalResult ReturnOp::verify() {
  FuncOp func = getOperation()->getParentOfType<FuncOp>();
  if (!func) {
    return emitOpError("must be nested directly in a tvm_ffi.func");
  }

  const mlir::FunctionType functionType = func.getFunctionType();
  if (functionType.getNumResults() != 1 ||
      !mlir::isa<AnyType, UnionType>(functionType.getResult(0))) {
    return mlir::success();
  }

  if (getNumOperands() == 0) {
    return emitOpError(
        "an any or union function must return at least one value");
  }
  mlir::Type const resultType = functionType.getResult(0);
  for (const mlir::Value operand : getOperands()) {
    if (!mlir::isa<TVMFFIABIType>(operand.getType())) {
      return emitOpError("operand type must have a TVMFFIAny ABI: ")
             << operand.getType();
    }
    if (const UnionType resultUnion = mlir::dyn_cast<UnionType>(resultType);
        resultUnion && !resultUnion.contains(operand.getType())) {
      return emitOpError("operand type is not a member of the result type: ")
             << operand.getType() << " vs " << resultType;
    }
  }
  return mlir::success();
}

mlir::LogicalResult ConstantDTypeOp::verify() {
  if (mlir::ArrayAttr values = getValue();
      values.size() == 3 &&
      llvm::all_of(values, [](mlir::Attribute value) -> bool {
        return mlir::isa<mlir::IntegerAttr>(value);
      })) {
    return mlir::success();
  }
  return emitOpError("dtype result requires [code, bits, lanes]");
}

mlir::LogicalResult ToOp::verify() {
  mlir::Type const nativeType = getValue().getType();
  mlir::Type const resultType = getResult().getType();
  auto nativeTypeInterface =
      mlir::dyn_cast<TVMFFINativeTypeInterface>(resultType);
  if (nativeTypeInterface &&
      nativeTypeInterface.getNativeType() == nativeType) {
    return mlir::success();
  }
  return emitOpError("unsupported native-to-TVM FFI conversion from ")
         << nativeType << " to " << resultType;
}

mlir::LogicalResult ToOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location>,
    mlir::ValueRange operands, mlir::DictionaryAttr, mlir::OpaqueProperties,
    mlir::RegionRange, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  if (operands.size() != 1) {
    return mlir::failure();
  }
  mlir::Type const nativeType = operands.front().getType();
  mlir::Type const resultType =
      llvm::TypeSwitch<mlir::Type, mlir::Type>(nativeType)
          .Case<mlir::IntegerType>(
              [context](mlir::IntegerType type) -> mlir::Type {
                if (type.getWidth() == 1) {
                  return BoolType::get(context);
                }
                if (type.getWidth() == 64) {
                  return IntType::get(context);
                }
                return {};
              })
          .Case<mlir::Float64Type>([context](mlir::Float64Type) -> mlir::Type {
            return FloatType::get(context);
          })
          .Default([](mlir::Type) -> mlir::Type { return {}; });
  if (!resultType) {
    return mlir::failure();
  }
  inferredReturnTypes.push_back(resultType);
  return mlir::success();
}

mlir::LogicalResult TensorLiteralOp::verify() {
  if (!mlir::isa<mlir::DenseElementsAttr>(getValue())) {
    return emitOpError("requires a dense elements attribute");
  }
  mlir::RankedTensorType const type =
      mlir::dyn_cast<mlir::RankedTensorType>(getValue().getType());
  if (!type || !type.hasStaticShape()) {
    return emitOpError("requires a statically shaped ranked tensor literal");
  }
  if (!mlir::isa<mlir::IntegerType, mlir::FloatType>(type.getElementType())) {
    return emitOpError(
        "requires an integer, boolean, or floating-point element type");
  }
  return mlir::success();
}

mlir::LogicalResult GetOp::verify() {
  mlir::Type const tvmFFIType = getOperand().getType();
  mlir::Type const resultType = getResult().getType();
  if (mlir::isa<mlir::torch::Torch::BaseTensorType>(tvmFFIType) &&
      mlir::isa<ObjectType>(resultType)) {
    return mlir::success();
  }
  if (auto nativeTypeInterface =
          mlir::dyn_cast<TVMFFINativeTypeInterface>(tvmFFIType);
      nativeTypeInterface &&
      nativeTypeInterface.getNativeType() == resultType) {
    return mlir::success();
  }
  return emitOpError("unsupported get from ")
         << tvmFFIType << " to " << resultType;
}

mlir::LogicalResult GetOp::inferReturnTypes(
    mlir::MLIRContext *, std::optional<mlir::Location>,
    mlir::ValueRange operands, mlir::DictionaryAttr, mlir::OpaqueProperties,
    mlir::RegionRange, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  if (operands.size() != 1) {
    return mlir::failure();
  }
  mlir::Type const inputType = operands.front().getType();
  if (mlir::isa<mlir::torch::Torch::BaseTensorType>(inputType)) {
    inferredReturnTypes.push_back(ObjectType::get(inputType.getContext()));
    return mlir::success();
  }
  auto nativeTypeInterface =
      mlir::dyn_cast<TVMFFINativeTypeInterface>(inputType);
  if (!nativeTypeInterface) {
    return mlir::failure();
  }
  inferredReturnTypes.push_back(nativeTypeInterface.getNativeType());
  return mlir::success();
}

mlir::LogicalResult AsOp::verify() {
  if (auto resultType = getResult().getType();
      !mlir::isa<dlpack::DLTensorType>(resultType)) {
    return emitOpError("unsupported object view type ") << resultType;
  }
  return mlir::success();
}

} // namespace trident::tvm_ffi
