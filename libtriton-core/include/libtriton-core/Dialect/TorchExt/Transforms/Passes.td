include "mlir/Pass/PassBase.td"

def NormalizeTorchExtOperands
    : Pass<"torchext-normalize-operands", "mlir::func::FuncOp"> {
  let summary =
      "Canonicalize TorchExt launch operands to builtin tensor values";
  let description =
      [{Rewrites torch_ext.triton_kernel_launch operands produced by torch_c
            .to_builtin_tensor to use the underlying builtin tensor values.}];
  let dependentDialects = [
    "libtriton::torch_ext::TorchExtDialect", "mlir::func::FuncDialect",
    "mlir::torch::TorchConversion::TorchConversionDialect"
  ];
}

def AsyncKernelLaunch : Pass<"torchext-async-kernel-launch", "mlir::ModuleOp"> {
  let summary =
      "Ensure kernel launch ops carry an async stream operand";
  let description = [{
    Rewrites gpu.launch_func and torch_ext.triton_kernel_launch in a module.
    If the optional async/stream operand is not provided, this pass materializes
    torch_ext.get_current_stream and wires the resulting stream into the launch
    op.
  }];
  let dependentDialects = [
    "libtriton::torch_ext::TorchExtDialect", "mlir::gpu::GPUDialect"
  ];
}
