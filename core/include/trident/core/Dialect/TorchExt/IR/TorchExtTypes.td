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

// Legacy dtype attribute wrapper; runtime values use !tvm_ffi.dtype.
def TorchExt_DTypeType : TypeDef<TorchExt_Dialect, "DType"> {
  let mnemonic = "dtype";
}

#endif // TRIDENT_CORE_DIALECT_TORCHEXT_IR_TORCHEXTTYPES_TD_
