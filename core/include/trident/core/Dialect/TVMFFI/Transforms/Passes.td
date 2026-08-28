//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def DecomposeTVMFFI : Pass<"decompose-tvm-ffi", "mlir::ModuleOp"> {
  let summary = "Decompose composite TVMFFI operations";
  let description = [{Decompose TVMFFI array operations into function calls.}];
  let dependentDialects = ["trident::tvm_ffi::TVMFFIDialect"];
}

def ApplyObjectOwnership : Pass<"apply-object-ownership", "mlir::ModuleOp"> {
  let summary = "Materialize TVM FFI object ownership operations";
  let description = [{
    Analyze object ownership and insert the required ObjectIncRef and
    ObjectDecRef operations before region terminators.
  }];
  let dependentDialects = ["trident::tvm_ffi::TVMFFIDialect"];
}
