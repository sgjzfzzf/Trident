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
#include <torch-mlir/Dialect/Torch/IR/TorchDialect.h>
#include <torch-mlir/Dialect/Torch/IR/TorchTypes.h>

namespace trident::tvm_ffi {

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
  mlir::Type input = inputs[0];
  mlir::Type output = outputs[0];
  return mlir::isa<AnyType, UnionType>(output) &&
         (mlir::isa<mlir::torch::Torch::TorchDialect>(input.getDialect()) ||
          (input.hasTrait<mlir::TypeTrait::TVMFFIABI>() &&
           (mlir::isa<AnyType>(output) ||
            mlir::cast<UnionType>(output).contains(input))));
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

} // namespace trident::tvm_ffi
