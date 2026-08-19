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
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIAttributes.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.cpp.inc"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_ATTRDEF_CLASSES
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.cpp.inc"

#define GET_OP_CLASSES
#include "trident/core/Dialect/TVMFFI/IR/TVMFFI.cpp.inc"

namespace trident::tvm_ffi {

void TVMFFIDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIAttributes.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "trident/core/Dialect/TVMFFI/IR/TVMFFI.cpp.inc"
      >();
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

mlir::LogicalResult ConstantOp::verify() {
  mlir::Type type = getResult().getType();
  if (mlir::isa<BoolType>(type) && !mlir::isa<mlir::BoolAttr>(getValue())) {
    return emitOpError("bool result requires a BoolAttr");
  } else if (mlir::isa<IntType>(type) &&
             !mlir::isa<mlir::IntegerAttr>(getValue())) {
    return emitOpError("int result requires an IntegerAttr");
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
  if (!mlir::isa<ArrayType>(base)) {
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
