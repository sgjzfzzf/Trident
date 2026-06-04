include "mlir/Pass/PassBase.td"

def FinalizeTVMFFICall : Pass<"finalize-tvm-ffi-call", "mlir::ModuleOp"> {
  let summary = "Finalize tvm_ffi.call by resolving function names to handles";
  // clang-format off
  let description = [{
    Finds all `tvm_ffi.call` ops and resolves their string function names to
    `!tvm_ffi.object_handle` values obtained via `tvm_ffi.function_get_global`.
    For each unique function name, the pass creates:

    - A module-level `llvm.mlir.global` of type `!tvm_ffi.object_handle` to
      cache the resolved handle.
    - An init function that calls `tvm_ffi.function_get_global` and stores
      the result into the global.
    - A destroy function that calls `tvm_ffi.object_dec_ref` on the handle.
    - `llvm.mlir.global_ctors` / `llvm.mlir.global_dtors` to register the
      init/destroy for automatic execution at module load/unload time.

    Each `tvm_ffi.call` op is then replaced with a load from the global
    variable followed by a `tvm_ffi.function_call`.
  }];
  // clang-format on
  let dependentDialects =
      ["mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect"];
  let options = [
    Option<"initName", "init-name", "std::string",
           /*default=*/"\"__tvm_ffi_init\"",
           "Name of the module constructor function">,
    Option<"finiName", "fini-name", "std::string",
           /*default=*/"\"__tvm_ffi_fini\"",
           "Name of the module destructor function">,
  ];
}
