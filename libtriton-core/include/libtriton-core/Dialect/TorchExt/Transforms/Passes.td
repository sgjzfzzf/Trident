#ifndef LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_PASSES_TD_
#define LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def RewriteTorchAsTorchExt
    : Pass<"rewrite-torch-as-torchext", "mlir::ModuleOp"> {
  let summary = "Rewrite Torch dialect aten ops as TorchExt dispatcher calls";
  // clang-format off
  let description = [{
    Rewrites Torch dialect ops whose names start with
    "torch.aten."(e.g. torch.aten.empty_like) into TorchExt
    torchext.call_dispatcher ops.  The  "aten."  prefix becomes
    "aten::" as the dispatcher op_name; the overload_name is left empty.
  }];
  // clang-format on
  let dependentDialects = ["libtriton::torchext::TorchExtDialect"];
}

#endif // LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_PASSES_TD_
