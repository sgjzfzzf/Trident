//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TORCHEXT_IR_TORCHEXTATTRS_TD_
#define TRIDENT_CORE_DIALECT_TORCHEXT_IR_TORCHEXTATTRS_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "trident/core/Dialect/TorchExt/IR/TorchExtInterfaces.td"

def TorchExt_SpecializationArrayAttr
    : TypedArrayAttrBase<TorchExt_SpecializationAttrInterface,
                         "array of Triton argument specializations">;

def TorchExt_ConstantSpecializationAttr
    : AttrDef<TorchExt_Dialect, "ConstantSpecialization", [
        DeclareAttrInterfaceMethods<TorchExt_SpecializationAttrInterface>
      ]> {
  let mnemonic = "constant_specialization";
  let summary = "Triton constant specialization assumption";
  let parameters = (ins "::mlir::Attribute":$value);
  let assemblyFormat = [{ `<` `value` `=` $value `>` }];
  let genVerifyDecl = 1;
}

def TorchExt_VariableSpecializationAttr
    : AttrDef<TorchExt_Dialect, "VariableSpecialization", [
        DeclareAttrInterfaceMethods<TorchExt_SpecializationAttrInterface>
      ]> {
  let mnemonic = "variable_specialization";
  let summary = "Triton variable specialization assumption";
  let parameters = (ins
    "::mlir::TypeAttr":$kind,
    DefaultValuedParameter<"uint64_t", "1">:$divisibility
  );
  let assemblyFormat = [{
    `<` `kind` `=` $kind (`,` `divisibility` `=` $divisibility^)? `>`
  }];
}

#endif // TRIDENT_CORE_DIALECT_TORCHEXT_IR_TORCHEXTATTRS_TD_
