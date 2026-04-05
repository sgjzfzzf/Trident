#ifndef LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIBASE_TD_
#define LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIBASE_TD_

include "mlir/IR/DialectBase.td"

def TVMFFI_Dialect : Dialect {
  let name = "tvm_ffi";
  let cppNamespace = "::libtriton::tvm_ffi";
  let useDefaultTypePrinterParser = 1;
  let summary = "TVM runtime FFI modeling dialect";
  let description =
      [{The TVMFFI dialect models value conversions between MLIR -
        level values and runtime FFI -
        compatible handles used by the TVM runtime boundary
            .

        At this stage the dialect primarily provides conversion scaffold
            operations and custom types used by lowering passes.}];
}

#endif // LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIBASE_TD_