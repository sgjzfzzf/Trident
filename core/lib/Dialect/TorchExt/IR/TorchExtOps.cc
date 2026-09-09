//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/Torch/IR/TorchInterfaces.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtTypes.h"
#include <cassert>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <optional>
#include <torch-mlir/Dialect/Torch/IR/TorchTypes.h>

namespace trident::torchext {

bool CastOp::areCastCompatible(mlir::TypeRange inputs,
                               mlir::TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    return false;
  }
  mlir::Type const input = inputs[0];
  mlir::Type const output = outputs[0];
  if (mlir::isa<mlir::torch::Torch::AnyType>(output)) {
    return true;
  }
  mlir::torch::Torch::UnionType const unionType =
      mlir::dyn_cast<mlir::torch::Torch::UnionType>(output);
  return unionType && llvm::is_contained(unionType.getContainedTypes(), input);
}

mlir::ParseResult TritonKernelLaunchOp::parseKernelArguments(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &operands,
    llvm::SmallVectorImpl<mlir::Type> &types,
    mlir::ArrayAttr &specializations) {
  llvm::SmallVector<mlir::Attribute> parsedSpecializations;

  do {
    mlir::OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand) || parser.parseColon()) {
      return mlir::failure();
    }

    mlir::Type type;
    if (parser.parseType(type)) {
      return mlir::failure();
    }

    mlir::Attribute specialization;
    if (parser.parseAttribute(specialization)) {
      return mlir::failure();
    }
    parsedSpecializations.push_back(specialization);
    operands.push_back(operand);
    types.push_back(type);
  } while (mlir::succeeded(parser.parseOptionalComma()));

  specializations =
      mlir::ArrayAttr::get(parser.getContext(), parsedSpecializations);
  return mlir::success();
}

void TritonKernelLaunchOp::printKernelArguments(
    mlir::OpAsmPrinter &printer, mlir::Operation *op [[maybe_unused]],
    mlir::OperandRange operands, mlir::TypeRange types,
    mlir::ArrayAttr specializations) {
  assert(operands.size() == types.size());
  assert(specializations.size() == operands.size());

  llvm::interleaveComma(llvm::enumerate(llvm::zip(operands, types)),
                        printer.getStream(), [&](auto indexedOperandAndType) {
                          auto [index, operandsAndType] = indexedOperandAndType;
                          auto [operand, type] = operandsAndType;
                          printer.printOperand(operand);
                          printer << " : ";
                          printer.printType(type);
                          printer << ' ';
                          printer.printAttribute(specializations[index]);
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

mlir::LogicalResult GetOp::inferReturnTypes(
    mlir::MLIRContext *, std::optional<mlir::Location>,
    mlir::ValueRange operands, mlir::DictionaryAttr, mlir::OpaqueProperties,
    mlir::RegionRange, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  if (operands.size() != 1) {
    return mlir::failure();
  }
  mlir::Type inputType = operands.front().getType();
  if (auto torchTypeInterface =
          mlir::dyn_cast<trident::torch::TorchToTVMFFITypeInterface>(
              inputType)) {
    inputType = torchTypeInterface.getTVMFFIType();
  }
  auto nativeTypeInterface =
      mlir::dyn_cast<tvm_ffi::TVMFFINativeTypeInterface>(inputType);
  if (!nativeTypeInterface) {
    return mlir::failure();
  }
  inferredReturnTypes.push_back(nativeTypeInterface.getNativeType());
  return mlir::success();
}

mlir::LogicalResult TritonKernelLaunchOp::verify() {
  if (getSpecializations().size() != getKernelOperands().size()) {
    return emitOpError(
        "specializations and kernel operands must have the same size");
  }
  return mlir::success();
}

} // namespace trident::torchext
