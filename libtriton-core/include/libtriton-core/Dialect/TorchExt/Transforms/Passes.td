include "mlir/Pass/PassBase.td"

def NormalizeTorchExtOperands
    : Pass<"torchext-normalize-operands", "mlir::func::FuncOp"> {
  let summary =
      "Canonicalize TorchExt launch operands to builtin tensor values";
  let description =
      [{Rewrites torch_ext.torch_kernel_launch operands produced by torch_c
            .to_builtin_tensor to use the underlying builtin tensor values.}];
  let dependentDialects = [
    "libtriton::torch_ext::TorchExtDialect", "mlir::func::FuncDialect",
    "mlir::torch::TorchConversion::TorchConversionDialect"
  ];
}
