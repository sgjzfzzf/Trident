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
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIAttributes.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "llvm/ADT/TypeSwitch.h"

#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "trident/core/Dialect/TVMFFI/IR/TVMFFI.cpp.inc"

namespace trident::tvm_ffi {

void TVMFFIDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIAttributes.cpp.inc"
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
  if (!llvm::all_of(llvm::concat<const mlir::Type>(functionType.getInputs(),
                                                   functionType.getResults()),
                    [](mlir::Type type) {
                      return mlir::isa<mlir::torch::Torch::TorchDialect>(
                          type.getDialect());
                    })) {
    return emitOpError("all inputs and outputs must be Torch dialect types");
  }

  if (std::optional<llvm::StringRef> visibility = getSymVisibility();
      visibility && *visibility != "public") {
    return emitOpError("must have public visibility");
  }

  return mlir::success();
}

} // namespace trident::tvm_ffi
