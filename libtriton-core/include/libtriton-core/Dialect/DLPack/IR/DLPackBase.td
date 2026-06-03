#ifndef LIBTRITON_CORE_DIALECT_DLPACK_IR_DLPACKBASE_TD_
#define LIBTRITON_CORE_DIALECT_DLPACK_IR_DLPACKBASE_TD_

include "mlir/IR/DialectBase.td"

def DLPack_Dialect : Dialect {
  let name = "dlpack";
  let cppNamespace = "::libtriton::dlpack";
  let useDefaultTypePrinterParser = 1;
  let summary = "DLPack modeling dialect";
  // clang-format off
  let description = [{
    The DLPack dialect models DLPack-compatible value types used at
    interoperability boundaries.
  }];
  // clang-format on
}

#endif // LIBTRITON_CORE_DIALECT_DLPACK_IR_DLPACKBASE_TD_