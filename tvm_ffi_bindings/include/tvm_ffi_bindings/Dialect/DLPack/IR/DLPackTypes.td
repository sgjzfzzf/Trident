#ifndef TVM_FFI_BINDINGS_DIALECT_DLPACK_IR_DLPACKTYPES_TD_
#define TVM_FFI_BINDINGS_DIALECT_DLPACK_IR_DLPACKTYPES_TD_

include "mlir/IR/AttrTypeBase.td"
include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackBase.td"

class DLPack_Type<string name, string typeMnemonic>
    : TypeDef<DLPack_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def DLPack_DLContextType : DLPack_Type<"DLContext", "dlcontext"> {
  let summary = "DLPack DLContext value type";
  let description = [{Represents the legacy DLPack DLContext structure used by
                          the referenced dlpack.h ABI.}];
}

def DLPack_DLDataTypeType : DLPack_Type<"DLDataType", "dldatatype"> {
  let summary = "DLPack DLDataType value type";
  let description =
      [{Represents the DLPack DLDataType structure describing the element code,
        bit width, and lane count.}];
}

def DLPack_DLTensorType : DLPack_Type<"DLTensorType", "dltensor"> {
  let summary = "Opaque DLTensor handle type";
  let description = [{Represents a DLTensor -
     compatible handle value flowing across DLPack interoperability boundaries.}];
}

#endif // TVM_FFI_BINDINGS_DIALECT_DLPACK_IR_DLPACKTYPES_TD_
