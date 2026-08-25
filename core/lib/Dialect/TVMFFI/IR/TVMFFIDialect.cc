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
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.cpp.inc"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.cpp.inc"

#define GET_OP_CLASSES
#include "trident/core/Dialect/TVMFFI/IR/TVMFFI.cpp.inc"

namespace trident::tvm_ffi {

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
  if (inputs.size() != 1 || outputs.size() != 1 ||
      !mlir::isa<AnyType>(outputs.front())) {
    return false;
  }

  mlir::Type input = inputs.front();
  return input.hasTrait<::mlir::TypeTrait::AnyABI>() ||
         input.getDialect().getNamespace() == "torch";
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
                   llvm::ArrayRef<mlir::NamedAttribute> attrs,
                   mlir::ArrayAttr argAttrs, mlir::ArrayAttr resAttrs) {
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
  state.addAttribute(getArgAttrsAttrName(state.name), argAttrs);
  state.addAttribute(getResAttrsAttrName(state.name), resAttrs);
}

mlir::LogicalResult FuncOp::verify() {
  mlir::FunctionType functionType = getFunctionType();
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

  mlir::FunctionType functionType = func.getFunctionType();
  if (functionType.getNumResults() != 1 ||
      !mlir::isa<AnyType>(functionType.getResult(0))) {
    return mlir::success();
  }

  if (getNumOperands() == 0) {
    return emitOpError(
        "a !tvm_ffi.any function must return at least one value");
  }
  for (mlir::Value operand : getOperands()) {
    if (!operand.getType().hasTrait<mlir::TypeTrait::AnyABI>()) {
      return emitOpError("operand type must have a TVMFFIAny ABI: ")
             << operand.getType();
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
  } else if (type.hasTrait<mlir::TypeTrait::Object>()) {
    return emitOpError("object constants are not supported");
  } else {
    return mlir::success();
  }
}

mlir::LogicalResult ArrayGetItemOp::verify() {
  mlir::Type base = getArray().getType();
  if (!mlir::isa<ArrayType, mlir::torch::Torch::AnyType,
                 mlir::torch::Torch::ListType, mlir::torch::Torch::TupleType>(
          base)) {
    return emitOpError("array operand must be !tvm_ffi.array");
  } else if (!getElementType()) {
    return emitOpError("array element type must be specified");
  } else if (mlir::Type element = *getElementType();
             getResult().getType() != element) {
    return emitOpError("result type must be ") << element;
  } else {
    return mlir::success();
  }
}

} // namespace trident::tvm_ffi
