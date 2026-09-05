//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/Torch/IR/TorchInterfaces.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtAttrs.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtTypes.h"
#include <cassert>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <torch-mlir/Dialect/Torch/IR/TorchTypes.h>

namespace trident::torchext {

namespace {

constexpr llvm::StringLiteral kSpecializationName = "triton.specialization";

} // namespace

mlir::ParseResult TritonKernelLaunchOp::parseKernelArguments(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &operands,
    llvm::SmallVectorImpl<mlir::Type> &types, mlir::ArrayAttr &argAttrs) {
  llvm::SmallVector<mlir::Attribute> attributes;
  bool hasAttributes = false;

  do {
    mlir::OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand) || parser.parseColon()) {
      return mlir::failure();
    }

    mlir::Type type;
    if (parser.parseType(type)) {
      return mlir::failure();
    }

    mlir::NamedAttrList namedAttrs;
    if (parser.parseOptionalAttrDict(namedAttrs)) {
      return mlir::failure();
    }
    hasAttributes |= !namedAttrs.empty();
    attributes.push_back(namedAttrs.getDictionary(parser.getContext()));
    operands.push_back(operand);
    types.push_back(type);
  } while (mlir::succeeded(parser.parseOptionalComma()));

  if (hasAttributes) {
    argAttrs = mlir::ArrayAttr::get(parser.getContext(), attributes);
  }
  return mlir::success();
}

void TritonKernelLaunchOp::printKernelArguments(mlir::OpAsmPrinter &printer,
                                                mlir::Operation *op
                                                [[maybe_unused]],
                                                mlir::OperandRange operands,
                                                mlir::TypeRange types,
                                                mlir::ArrayAttr argAttrs) {
  assert(operands.size() == types.size());
  assert(!argAttrs || argAttrs.size() == operands.size());

  llvm::interleaveComma(llvm::enumerate(llvm::zip(operands, types)),
                        printer.getStream(), [&](auto indexedOperandAndType) {
                          auto [index, operandsAndType] = indexedOperandAndType;
                          auto [operand, type] = operandsAndType;
                          printer.printOperand(operand);
                          printer << " : ";
                          printer.printType(type);
                          if (!argAttrs) {
                            return;
                          }
                          auto attrs =
                              mlir::cast<mlir::DictionaryAttr>(argAttrs[index]);
                          if (!attrs.empty()) {
                            printer.printOptionalAttrDict(attrs.getValue());
                          }
                        });
}

mlir::LogicalResult ConvertOp::verify() {
  if (mlir::isa<DTypeType>(getOperand().getType()) &&
      mlir::isa<mlir::torch::Torch::IntType>(getResult().getType())) {
    return mlir::success();
  }
  return emitOpError("expects !torchext.dtype -> !torch.int");
}

mlir::LogicalResult GetOp::verify() {
  mlir::Type const input = getOperand().getType();
  mlir::Type const output = getResult().getType();
  mlir::Type tvmFFIType = input;
  if (auto torchTypeInterface =
          mlir::dyn_cast<trident::torch::TorchToTVMFFITypeInterface>(input)) {
    tvmFFIType = torchTypeInterface.getTVMFFIType();
  }

  auto nativeTypeInterface =
      mlir::dyn_cast<tvm_ffi::TVMFFINativeTypeInterface>(tvmFFIType);
  if (!nativeTypeInterface) {
    return emitOpError("unsupported get from ") << input << " to " << output;
  }

  mlir::Type const nativeType = nativeTypeInterface.getNativeType();
  if (nativeType == output) {
    return mlir::success();
  }
  return emitOpError("unsupported get from ") << input << " to " << output;
}

mlir::LogicalResult TritonKernelLaunchOp::verify() {
  mlir::ArrayAttr const argAttrs = getArgAttrsAttr();
  if (!argAttrs) {
    return emitOpError("expects arg_attrs for every kernel operand");
  }
  if (argAttrs.size() != getKernelOperands().size()) {
    return emitOpError("arg_attrs and kernel operands must have the same size");
  }
  for (auto [index, argAttr, operand] : llvm::enumerate(
           argAttrs.getAsRange<mlir::DictionaryAttr>(), getKernelOperands())) {
    SpecializationAttr const specialization =
        mlir::dyn_cast_or_null<SpecializationAttr>(
            argAttr.get(kSpecializationName));
    if (!specialization) {
      return emitOpError("kernel operand #")
             << index << " requires a " << kSpecializationName << " attribute";
    }
    mlir::Type const operandType = operand.getType();
    mlir::Type const kind = specialization.getKind().getValue();
    auto integerKind = mlir::dyn_cast<mlir::IntegerType>(kind);
    bool const validKind =
        (mlir::isa<mlir::torch::Torch::BaseTensorType>(operandType) &&
         mlir::isa<mlir::LLVM::LLVMPointerType>(kind)) ||
        (mlir::isa<mlir::torch::Torch::BoolType, mlir::torch::Torch::IntType>(
             operandType) &&
         integerKind && integerKind.getWidth() <= 64) ||
        (mlir::isa<mlir::torch::Torch::FloatType>(operandType) &&
         (kind.isF32() || kind.isF64()));
    if (!validKind) {
      return emitOpError("kernel operand #")
             << index << " of type " << operandType
             << " cannot be converted to specialization kind " << kind;
    }
  }
  return mlir::success();
}

} // namespace trident::torchext
