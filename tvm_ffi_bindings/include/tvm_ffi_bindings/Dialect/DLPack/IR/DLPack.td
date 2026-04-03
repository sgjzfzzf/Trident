#ifndef TVM_FFI_BINDINGS_DIALECT_DLPACK_IR_DLPACK_TD_
#define TVM_FFI_BINDINGS_DIALECT_DLPACK_IR_DLPACK_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def DLPack_Dialect : Dialect {
  let name = "dlpack";
  let cppNamespace = "::libtriton::dlpack";
  let useDefaultTypePrinterParser = 1;
  let summary = "DLPack modeling dialect";
  let description =
      [{The DLPack dialect models DLPack -
        compatible value types used at interoperability boundaries.}];
}

include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackTypes.td"

#endif // TVM_FFI_BINDINGS_DIALECT_DLPACK_IR_DLPACK_TD_
