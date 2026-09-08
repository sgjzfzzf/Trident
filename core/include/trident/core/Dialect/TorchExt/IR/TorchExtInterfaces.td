//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TORCHEXT_IR_TORCHEXTINTERFACES_TD_
#define TRIDENT_CORE_DIALECT_TORCHEXT_IR_TORCHEXTINTERFACES_TD_

include "mlir/IR/OpBase.td"

def TorchExt_DTypeAttr : AttrInterface<"DTypeAttrInterface"> {
  let cppNamespace = "::trident::torchext";
  let description = [{
    Describes a Torch dtype and its DLPack representation.
  }];
  let methods = [
    InterfaceMethod<
      "Returns the corresponding DLPack dtype.", "::DLDataType",
      "getDLPackDType", (ins)
    >
  ];
}

def TorchExt_SpecializationAttrInterface
    : AttrInterface<"SpecializationAttrInterface"> {
  let cppNamespace = "::trident::torchext";
  let description = [{
    Describes a Triton argument specialization and materializes its runtime
    precondition when one is required.
  }];
  let methods = [
    InterfaceMethod<
      "Builds the native specialization check, or returns a null value when "
      "the specialization has no runtime precondition.",
      "::mlir::Value", "buildCheck",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc,
           "::mlir::Value":$operand)
    >,
    InterfaceMethod<
      "Returns the native type used to evaluate this specialization.",
      "::mlir::Type", "getTargetType", (ins)
    >
  ];
}

#endif // TRIDENT_CORE_DIALECT_TORCHEXT_IR_TORCHEXTINTERFACES_TD_
