include "mlir/Pass/PassBase.td"

def EmitTVMFFIInterface :
    Pass<"emit-tvm-ffi-interface", "mlir::ModuleOp"> {
  let summary = "Generate TVM stable C API wrappers for selected functions";
  let description = [{
    Generate TVM stable C API wrappers for selected functions.
  }];
  let dependentDialects = [
    "libtriton::dlpack::DLPackDialect",
    "libtriton::tvm_ffi::TVMFFIDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

def ConvertTVMFFIToLLVM :
    Pass<"convert-tvm-ffi-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower TVMFFI dialect operations to LLVM dialect";
  let description = [{
    Lower TVMFFI dialect operations to LLVM dialect.
  }];
}
