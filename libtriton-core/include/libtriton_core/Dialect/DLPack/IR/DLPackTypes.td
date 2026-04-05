#ifndef LIBTRITON_CORE_DIALECT_DLPACK_IR_DLPACKTYPES_TD_
#define LIBTRITON_CORE_DIALECT_DLPACK_IR_DLPACKTYPES_TD_

include "mlir/IR/AttrTypeBase.td"
include "libtriton_core/Dialect/DLPack/IR/DLPackBase.td"

class DLPack_Type<string name, string typeMnemonic>
    : TypeDef<DLPack_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def DLPack_DLContextType : DLPack_Type<"DLContext", "context"> {
  let summary = "DLPack DLContext value type";
  let description = [{Represents the legacy DLPack DLContext structure used by
                          the referenced dlpack.h ABI.}];
}

def DLPack_DLDataTypeType : DLPack_Type<"DLDataType", "datatype"> {
  let summary = "DLPack DLDataType value type";
  let description =
      [{Represents the DLPack DLDataType structure describing the element code,
        bit width, and lane count.}];
}

def DLPack_DLTensor : DLPack_Type<"DLTensor", "tensor"> {
  let summary = "Opaque DLTensor handle type";
  let description =
      "Represents a DLTensor-compatible handle value flowing across DLPack "
      "interoperability boundaries.";
}

def DLPack_DLManagedTensor : DLPack_Type<"DLManagedTensor", "mtensor"> {
  let summary = "Opaque DLManagedTensor handle type";
  let description =
      "Represents a DLManagedTensor-compatible handle value flowing across "
      "DLPack interoperability boundaries.";
}

#endif // LIBTRITON_CORE_DIALECT_DLPACK_IR_DLPACKTYPES_TD_
