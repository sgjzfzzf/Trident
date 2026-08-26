//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// TVMFFIDialect.cc - TVMFFI Dialect registration and initialization.
//
// This file implements the dialect `initialize()` method, which registers all
// ops and types defined in TVMFFI.td via ODS-generated .cpp.inc files.
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.cpp.inc"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtTypes.h" // NOLINT(misc-include-cleaner)
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/TypeSwitch.h> // NOLINT(misc-include-cleaner)
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h> // NOLINT(misc-include-cleaner)
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/Support/LLVM.h>
#include <optional>
#include <string>
#include <torch-mlir/Dialect/Torch/IR/TorchTypes.h>

namespace trident::tvm_ffi {

namespace {

static std::optional<int32_t> getConcreteTypeIndex(mlir::Type type) {
  return llvm::TypeSwitch<mlir::Type, std::optional<int32_t>>(type)
      .Case<ArrayType>([](ArrayType) { return ArrayType::getTypeIndex(); })
      .Case<BoolType>([](BoolType) { return BoolType::getTypeIndex(); })
      .Case<DeviceType>([](DeviceType) { return DeviceType::getTypeIndex(); })
      .Case<DTypeType>([](DTypeType) { return DTypeType::getTypeIndex(); })
      .Case<ExceptionType>(
          [](ExceptionType) { return ExceptionType::getTypeIndex(); })
      .Case<FloatType>([](FloatType) { return FloatType::getTypeIndex(); })
      .Case<FunctionType>(
          [](FunctionType) { return FunctionType::getTypeIndex(); })
      .Case<IntType>([](IntType) { return IntType::getTypeIndex(); })
      .Case<NoneType>([](NoneType) { return NoneType::getTypeIndex(); })
      .Case<RawStrType>([](RawStrType) { return RawStrType::getTypeIndex(); })
      .Case<SmallStrType>(
          [](SmallStrType) { return SmallStrType::getTypeIndex(); })
      .Case<StrType>([](StrType) { return StrType::getTypeIndex(); })
      .Case<TensorType>([](TensorType) { return TensorType::getTypeIndex(); })
      .Default([](mlir::Type) { return std::nullopt; });
}

} // namespace
} // namespace trident::tvm_ffi

#define GET_TYPEDEF_CLASSES
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.cpp.inc"

#define GET_OP_CLASSES
#include "trident/core/Dialect/TVMFFI/IR/TVMFFI.cpp.inc"

namespace trident::tvm_ffi {

mlir::LogicalResult
UnionType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                  llvm::ArrayRef<mlir::Type> types) {
  if (types.size() < 2) {
    return emitError() << "union requires at least two member types";
  }
  llvm::SmallDenseSet<mlir::Type> uniqueTypes;
  for (mlir::Type type : types) {
    if (!type.hasTrait<mlir::TypeTrait::TVMFFIABI>() ||
        mlir::isa<AnyType, UnionType>(type)) {
      return emitError() << "invalid union member type " << type;
    }
    if (!getConcreteTypeIndex(type)) {
      return emitError() << "invalid union member type " << type;
    }
    if (!uniqueTypes.insert(type).second) {
      return emitError() << "union member types must be unique";
    }
  }
  return mlir::success();
}

bool UnionType::contains(mlir::Type type) const {
  if (const UnionType unionType = mlir::dyn_cast<UnionType>(type)) {
    return llvm::all_of(unionType.getTypes(), [&](const mlir::Type member) {
      return llvm::is_contained(getTypes(), member);
    });
  }
  return llvm::is_contained(getTypes(), type);
}

void TVMFFIDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "trident/core/Dialect/TVMFFI/IR/TVMFFI.cpp.inc"
      >();
}

bool CastOp::areCastCompatible(mlir::TypeRange inputs,
                               mlir::TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    return false;
  }

  mlir::Type input = inputs.front();
  mlir::Type output = outputs.front();
  if (!mlir::isa<AnyType, UnionType>(output)) {
    return false;
  }
  if (input.getDialect().getNamespace() == "torch") {
    return mlir::isa<AnyType, UnionType>(output);
  }
  if (!input.hasTrait<mlir::TypeTrait::TVMFFIABI>()) {
    return false;
  }
  return mlir::isa<AnyType>(output) ||
         mlir::cast<UnionType>(output).contains(input);
}

// FuncOp custom assembly format.

mlir::Region *FuncOp::getCallableRegion() {
  mlir::Region &body = getBody();
  return body.empty() ? nullptr : &body;
}

llvm::ArrayRef<mlir::Type> FuncOp::getArgumentTypes() {
  return getFunctionType().getInputs();
}

llvm::ArrayRef<mlir::Type> FuncOp::getResultTypes() {
  return getFunctionType().getResults();
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name),
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); },
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::LogicalResult FuncOp::verify() {
  const mlir::FunctionType functionType = getFunctionType();
  // Existing frontends initially construct this wrapper with Torch types.
  // They are a supported input form; conversion makes the ownership-bearing
  // TVM FFI types explicit before runtime operations are lowered.

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
    if (!operand.getType().hasTrait<mlir::TypeTrait::TVMFFIABI>()) {
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
    } else {
      return emitOpError("dtype result requires [code, bits, lanes]");
    }
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
  } else {
    return mlir::success();
  }
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

} // namespace trident::tvm_ffi
