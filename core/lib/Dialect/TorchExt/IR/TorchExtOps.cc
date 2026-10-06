//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TorchExt/IR/TorchExtOps.h"
#include <cassert>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

namespace trident::torchext {

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

mlir::LogicalResult TritonKernelLaunchOp::verify() {
  if (getSpecializations().size() != getKernelOperands().size()) {
    return emitOpError(
        "specializations and kernel operands must have the same size");
  }
  return mlir::success();
}

} // namespace trident::torchext
