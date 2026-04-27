include "mlir/Pass/PassBase.td"

def EmitTVMFFIInterface : Pass<"emit-tvm-ffi-interface", "mlir::ModuleOp"> {
  let summary = "Generate TVM stable C API wrappers for selected functions";
  let description =
      [{ Generate TVM stable C API wrappers for selected functions. }];
  let dependentDialects = [
    "libtriton::dlpack::DLPackDialect", "libtriton::tvm_ffi::TVMFFIDialect",
    "mlir::arith::ArithDialect", "mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}
