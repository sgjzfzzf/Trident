#ifndef LIBTRITON_CORE_DIALECT_TORCHEXT_IR_TORCHEXTTYPES_TD_
#define LIBTRITON_CORE_DIALECT_TORCHEXT_IR_TORCHEXTTYPES_TD_

include "mlir/IR/AttrTypeBase.td"
include "libtriton-core/Dialect/TorchExt/IR/TorchExtBase.td"

class TorchExt_Type<string name, string typeMnemonic>
    : TypeDef<TorchExt_Dialect, name> {
  let mnemonic = typeMnemonic;
}

#endif // LIBTRITON_CORE_DIALECT_TORCHEXT_IR_TORCHEXTTYPES_TD_
