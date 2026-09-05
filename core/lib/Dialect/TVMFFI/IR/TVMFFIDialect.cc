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
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtTypes.h" // NOLINT(misc-include-cleaner)
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/TypeSwitch.h>             // NOLINT(misc-include-cleaner)
#include <mlir/Dialect/LLVMIR/LLVMDialect.h> // NOLINT(misc-include-cleaner)
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
#include <mlir/Transforms/InliningUtils.h>
#include <string>
#include <torch-mlir/Dialect/Torch/IR/TorchDialect.h>

#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.cpp.inc"

#define GET_OP_CLASSES
#include "trident/core/Dialect/TVMFFI/IR/TVMFFI.cpp.inc"

namespace trident::tvm_ffi {

namespace {

struct TVMFFIInlinerInterface final : mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(mlir::Operation *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }

  bool isLegalToInline(mlir::Region *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }
};

} // namespace

mlir::LogicalResult
UnionType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                  llvm::ArrayRef<mlir::Type> types) {
  if (types.size() < 2) {
    return emitError() << "union requires at least two member types";
  }
  llvm::SmallDenseSet<mlir::Type> uniqueTypes;
  for (mlir::Type const type : types) {
    if (!mlir::isa<TVMFFIABIType>(type) ||
        mlir::isa<AnyType, UnionType>(type)) {
      return emitError() << "invalid union member type " << type;
    }
    if (!mlir::isa<TVMFFITypeIndexInterface>(type)) {
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
    return llvm::all_of(unionType.getTypes(),
                        [&](const mlir::Type member) -> bool {
                          return llvm::is_contained(getTypes(), member);
                        });
  }
  return llvm::is_contained(getTypes(), type);
}

void TVMFFIDialect::initialize() {
  addInterfaces<TVMFFIInlinerInterface>();
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
  mlir::Type const input = inputs[0];
  mlir::Type output = outputs[0];
  return mlir::isa<AnyType, UnionType>(output) &&
         (mlir::isa<mlir::torch::Torch::TorchDialect>(input.getDialect()) ||
          (mlir::isa<TVMFFIABIType>(input) &&
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
