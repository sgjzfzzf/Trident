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

def TorchExt_SpecializationAttr
    : AttrDef<TorchExt_Dialect, "Specialization"> {
  let mnemonic = "specialization";
  let summary = "Triton kernel specialization assumption";
  let parameters = (ins
    "::mlir::TypeAttr":$kind,
    DefaultValuedParameter<"uint64_t", "1">:$divisibility
  );
  let assemblyFormat = [{
    `<` `kind` `=` $kind (`,` `divisibility` `=` $divisibility^)? `>`
  }];
}

#endif // TRIDENT_CORE_DIALECT_TORCHEXT_IR_TORCHEXTATTRS_TD_
