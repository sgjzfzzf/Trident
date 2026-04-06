#ifndef LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFITYPES_TD_
#define LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFITYPES_TD_

include "mlir/IR/AttrTypeBase.td"
include "libtriton_core/Dialect/TVMFFI/IR/TVMFFIBase.td"

class TVMFFI_Type<string name, string typeMnemonic>
    : TypeDef<TVMFFI_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def TVMFFI_AnyType : TVMFFI_Type<"Any", "any"> {
  let summary = "Type-erased TVM FFI value";
  let description =
      [{A generic, type - erased container type used at the TVM FFI boundary
                              .Conversion ops box typed MLIR values
                                  into this type and unbox them later.}];
}

def TVMFFI_ObjectHandleType : TVMFFI_Type<"ObjectHandle", "object_handle"> {
  let summary = "Opaque TVMFFI object handle";
  let description =
      [{Represents TVMFFIObjectHandle in the dialect type
            system.During LLVM lowering this type maps to an opaque pointer.}];
}

#endif // LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFITYPES_TD_
