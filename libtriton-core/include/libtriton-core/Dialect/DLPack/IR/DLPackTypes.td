#ifndef LIBTRITON_CORE_DIALECT_DLPACK_IR_DLPACKTYPES_TD_
#define LIBTRITON_CORE_DIALECT_DLPACK_IR_DLPACKTYPES_TD_

include "mlir/IR/AttrTypeBase.td"
include "libtriton-core/Dialect/DLPack/IR/DLPackBase.td"

class DLPack_Type<string name, string typeMnemonic>
    : TypeDef<DLPack_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def DLPack_DLDeviceType : DLPack_Type<"DLDevice", "device"> {
  let summary = "DLPack DLDevice value type";
  // clang-format off
  let description = [{
    Represents the DLPack DLDevice structure used by the referenced dlpack.h
    ABI.
  }];
  // clang-format on
}

def DLPack_DLDataTypeType : DLPack_Type<"DLDataType", "datatype"> {
  let summary = "DLPack DLDataType value type";
  // clang-format off
  let description =
      [{
        Represents the DLPack DLDataType structure describing the element
        code, bit width, and lane count.
      }];
  // clang-format on
}

def DLPack_DLTensor : DLPack_Type<"DLTensor", "tensor"> {
  let summary = "Opaque DLTensor handle type";
  // clang-format off
  let description = [{
    Represents a DLTensor-compatible handle value flowing across DLPack
    interoperability boundaries.
  }];
  // clang-format on
}

def DLPack_DLManagedTensor : DLPack_Type<"DLManagedTensor", "managed_tensor"> {
  let summary = "Opaque DLManagedTensor handle type";
  // clang-format off
  let description = [{
    Represents a DLManagedTensor-compatible handle value flowing across
    DLPack interoperability boundaries.
  }];
  // clang-format on
}

#endif // LIBTRITON_CORE_DIALECT_DLPACK_IR_DLPACKTYPES_TD_
