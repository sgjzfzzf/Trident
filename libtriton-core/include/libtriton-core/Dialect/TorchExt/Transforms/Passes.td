#ifndef LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_PASSES_TD_
#define LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def RAAI : Pass<"raai", "mlir::ModuleOp"> {
  let summary =
      "Release allocated intermediate variables in single-block regions";
  // clang-format off
  let description = [{
    Walks every op in the module.  For each single-block region (func::FuncOp,
    scf.if / scf.for bodies, etc.) it inserts torchext.ObjectIncRef for every
    yielded value and torchext.ObjectDecRef for every internally-allocated
    value (torch.prim.ListConstruct and torch.aten.* tensor/list results).
  }];
  // clang-format on
  let dependentDialects = ["libtriton::torchext::TorchExtDialect"];
}

#endif // LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_PASSES_TD_
