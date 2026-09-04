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

def OwnershipDeallocation : Pass<"ownership-deallocation", "mlir::ModuleOp"> {
  let summary = "Materialize TVM FFI object ownership in CFGs";
  let description = [{
    Analyze object ownership and liveness in function CFGs and insert the
    required ObjectIncRef and ObjectDecRef operations on exits and CFG edges.
  }];
  let dependentDialects = [
    "mlir::cf::ControlFlowDialect", "trident::tvm_ffi::TVMFFIDialect"
  ];
}
