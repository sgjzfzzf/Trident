//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def GeneralizeAtenOps : Pass<"generalize-aten-ops", "mlir::ModuleOp"> {
  let summary = "Represent concrete ATen operations as opaque Torch operators";
  let description = [{
    Rewrites concrete torch.aten operations to equivalent torch.operator
    operations so backend lowering does not invoke value-semantic ATen folders.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::torch::Torch::TorchDialect",
    "mlir::torch::TorchConversion::TorchConversionDialect",
    "trident::torchext::TorchExtDialect"
  ];
}
