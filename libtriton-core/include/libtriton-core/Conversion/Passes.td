include "mlir/Pass/PassBase.td"

def ConvertDLPackToLLVM : Pass<"convert-dlpack-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower DLPack dialect operations to LLVM dialect";
  let description = [{Lower DLPack dialect operations to LLVM dialect.}];
}

def ConvertTVMFFIToLLVM : Pass<"convert-tvm-ffi-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower TVMFFI dialect operations to LLVM dialect";
  let description = [{Lower TVMFFI dialect operations to LLVM dialect.}];
}

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

def ConvertTritonRTToLLVM : Pass<"convert-tritonrt-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower TritonRT launch op to GPU/LLVM-compatible form";
  let description =
      [{Rewrites triton_rt.triton_kernel_launch into
            gpu.launch_func.Kernel operands that are LLVM memref -
        descriptor -
        like structs are lowered to their aligned pointer field(index 1)
            .Other scalar -
        like operands are preserved
            .Two trailing null pointers are appended to satisfy Triton kernel
                ABI conventions.}];
  let dependentDialects = [
    "libtriton::triton_rt::TritonRTDialect", "mlir::func::FuncDialect",
    "mlir::gpu::GPUDialect", "mlir::LLVM::LLVMDialect"
  ];
}
