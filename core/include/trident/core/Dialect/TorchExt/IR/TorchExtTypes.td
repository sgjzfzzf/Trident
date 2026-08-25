//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TORCHEXT_IR_TORCHEXTTYPES_TD_
#define TRIDENT_CORE_DIALECT_TORCHEXT_IR_TORCHEXTTYPES_TD_

include "mlir/IR/AttrTypeBase.td"
include "trident/core/Dialect/TorchExt/IR/TorchExtBase.td"

// Semantic wrapper type for a torch.dtype value.  It is intentionally kept
// outside TorchMLIR's AnyTorchType set; wrapper lowering converts it to the
// native TVM FFI dtype ABI before converting it to Torch's ScalarType integer
// when calling a TorchMLIR Aten operation.
def TorchExt_DTypeType : TypeDef<TorchExt_Dialect, "DType"> {
  let mnemonic = "dtype";
}

#endif // TRIDENT_CORE_DIALECT_TORCHEXT_IR_TORCHEXTTYPES_TD_
