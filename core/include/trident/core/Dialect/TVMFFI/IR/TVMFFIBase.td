//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIBASE_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIBASE_TD_

include "mlir/IR/DialectBase.td"

def TVMFFI_Dialect : Dialect {
  let name = "tvm_ffi";
  let cppNamespace = "::trident::tvm_ffi";
  let summary = "TVMFFI dialect";
  let description = [{
    The TVMFFI dialect models the TVM runtime FFI (foreign function interface)
    for interoperability between TVM/PyTorch and Triton-compiled kernels.
    This dialect provides operations for managing TVM-packed arguments,
    invoking TVM runtime functions, and converting between TVM and native data
    types.
  }];

  let useDefaultAttributePrinterParser = 1;
}

#endif // TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIBASE_TD_
