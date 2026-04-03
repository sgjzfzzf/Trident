#ifndef TVM_FFI_BINDINGS_DIALECT_DLPACK_IR_DLPACKTYPES_TD_
#define TVM_FFI_BINDINGS_DIALECT_DLPACK_IR_DLPACKTYPES_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPack.td"

class DLPack_Type<string name, string typeMnemonic>
    : TypeDef<DLPack_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def DLPack_DLTensorType : DLPack_Type<"DLTensorType", "dltensor"> {
  let summary = "Opaque DLTensor handle type";
  let description = [{Represents a DLTensor -
     compatible handle value flowing across DLPack interoperability boundaries.}];
}

#endif // TVM_FFI_BINDINGS_DIALECT_DLPACK_IR_DLPACKTYPES_TD_
