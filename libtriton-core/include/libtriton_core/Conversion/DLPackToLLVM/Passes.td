include "mlir/Pass/PassBase.td"

def ConvertDLPackToLLVM :
    Pass<"convert-dlpack-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower DLPack dialect operations to LLVM dialect";
  let description = [{
    Lower DLPack dialect operations to LLVM dialect.
  }];
}
