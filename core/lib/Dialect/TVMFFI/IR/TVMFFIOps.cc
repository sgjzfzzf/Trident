//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
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

mlir::LogicalResult ConstantOp::verify() {
  mlir::Type type = getResult().getType();
  if (mlir::isa<BoolType>(type) && !mlir::isa<mlir::BoolAttr>(getValue())) {
    return emitOpError("bool result requires a BoolAttr");
  } else if (mlir::isa<IntType>(type) &&
             !mlir::isa<mlir::IntegerAttr>(getValue())) {
    return emitOpError("int result requires an IntegerAttr");
  } else if (mlir::isa<DTypeType>(type)) {
    if (mlir::ArrayAttr values = mlir::dyn_cast<mlir::ArrayAttr>(getValue());
        values && values.size() == 3 &&
        llvm::all_of(values, [](mlir::Attribute value) {
          return mlir::isa<mlir::IntegerAttr>(value);
        })) {
      return mlir::success();
    }
    return emitOpError("dtype result requires [code, bits, lanes]");
  } else if (mlir::isa<FloatType>(type) &&
             !mlir::isa<mlir::FloatAttr>(getValue())) {
    return emitOpError("float result requires a FloatAttr");
  } else if (mlir::isa<NoneType>(type) &&
             !mlir::isa<mlir::UnitAttr>(getValue())) {
    return emitOpError("none result requires a UnitAttr");
  } else if (mlir::isa<RawStrType>(type) &&
             !mlir::isa<mlir::StringAttr>(getValue())) {
    return emitOpError("string result requires a StringAttr");
  } else if (type.hasTrait<mlir::TypeTrait::Object>()) {
    return emitOpError("object constants are not supported");
  }
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

mlir::LogicalResult ArrayGetItemOp::verify() {
  const mlir::Type base = getArray().getType();
  if (!mlir::isa<ArrayType, mlir::torch::Torch::AnyType,
                 mlir::torch::Torch::ListType, mlir::torch::Torch::TupleType>(
          base)) {
    return emitOpError("array operand must be !tvm_ffi.array");
  }
  const std::optional<mlir::Type> element = getElementType();
  if (!element) {
    return emitOpError("array element type must be specified");
  }
  if (getResult().getType() != *element) {
    return emitOpError("result type must be ") << *element;
  }
  return mlir::success();
}

mlir::LogicalResult GetOp::verify() {
  mlir::Type const operandType = getOperand().getType();
  mlir::Type const resultType = getResult().getType();
  if (mlir::isa<BoolType>(operandType) && !resultType.isSignlessInteger(1)) {
    return emitOpError("a bool operand must produce i1");
  }
  if (mlir::isa<IntType>(operandType) &&
      (!mlir::isa<mlir::IntegerType>(resultType) ||
       resultType.getIntOrFloatBitWidth() > 64)) {
    return emitOpError("an int operand must produce an integer from i1 to i64");
  }
  if (mlir::isa<FloatType>(operandType) &&
      (!mlir::isa<mlir::FloatType>(resultType) ||
       resultType.getIntOrFloatBitWidth() > 64)) {
    return emitOpError(
        "a float operand must produce a float no wider than f64");
  }
  if (mlir::isa<TensorType>(operandType) &&
      !mlir::isa<mlir::LLVM::LLVMPointerType>(resultType)) {
    return emitOpError("a tensor operand must produce an LLVM pointer");
  }
  if (mlir::isa<AnyType>(operandType) &&
      !mlir::isa<mlir::IntegerType, mlir::FloatType,
                 mlir::LLVM::LLVMPointerType>(resultType)) {
    return emitOpError(
        "an any operand must produce a native scalar or pointer");
  }
  return mlir::success();
}

} // namespace trident::tvm_ffi
