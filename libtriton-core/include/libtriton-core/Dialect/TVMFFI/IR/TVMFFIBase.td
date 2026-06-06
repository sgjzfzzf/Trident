//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIBASE_TD_
#define LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIBASE_TD_

include "mlir/IR/DialectBase.td"

def TVMFFI_Dialect : Dialect {
  let name = "tvm_ffi";
  let cppNamespace = "::libtriton::tvm_ffi";
  let summary = "TVMFFI dialect";
  let description = [{
    The TVMFFI dialect models the TVM runtime FFI (foreign function interface)
    for interoperability between TVM/PyTorch and Triton-compiled kernels.
    This dialect provides operations for managing TVM-packed arguments,
    invoking TVM runtime functions, and converting between TVM and native data
    types.
  }];
}

#endif // LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIBASE_TD_
