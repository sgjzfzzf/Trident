//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def DecomposeSpecialization : Pass<"decompose-specialization", "mlir::ModuleOp"> {
  let summary = "Decompose Triton kernel specialization assumptions";
  let description = [{
    Materializes runtime checks for Triton divisibility assumptions attached to
    Triton kernel launch operations, then removes the specialization metadata.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect", "mlir::cf::ControlFlowDialect",
    "trident::dlpack::DLPackDialect", "mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect",
    "trident::torchext::TorchExtDialect", "trident::tvm_ffi::TVMFFIDialect"
  ];
}
