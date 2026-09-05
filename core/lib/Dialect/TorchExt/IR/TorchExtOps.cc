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
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>
#include <torch-mlir/Dialect/Torch/IR/TorchTypes.h>

namespace trident::torchext {

mlir::ParseResult TridentKernelLaunchOp::parseKernelArguments(
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

void TridentKernelLaunchOp::printKernelArguments(mlir::OpAsmPrinter &printer,
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

} // namespace trident::torchext
