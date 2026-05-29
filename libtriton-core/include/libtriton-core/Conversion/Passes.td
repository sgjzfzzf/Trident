include "mlir/Pass/PassBase.td"

def ConvertDLPackToLLVM : Pass<"convert-dlpack-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower DLPack dialect operations to LLVM dialect";
  let description = [{Lower DLPack dialect operations to LLVM dialect.}];
}

def ConvertTVMFFIToLLVM : Pass<"convert-tvm-ffi-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower TVMFFI dialect operations to LLVM dialect";
  let description = [{Lower TVMFFI dialect operations to LLVM dialect.}];
}

def ConvertTorchExtToLLVM : Pass<"convert-torchext-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower TorchExt launch op to GPU/LLVM-compatible form";
  let description =
      [{Rewrites torch_ext.triton_kernel_launch into
            gpu.launch_func.Kernel operands that are LLVM memref -
        descriptor -
        like structs are lowered to their aligned pointer field(index 1)
            .Other scalar -
        like operands are preserved
            .Two trailing null pointers are appended to satisfy Torch kernel ABI
                conventions.}];
  let dependentDialects = [
    "libtriton::torch_ext::TorchExtDialect", "mlir::func::FuncDialect",
    "mlir::gpu::GPUDialect", "mlir::LLVM::LLVMDialect"
  ];
}

def LibTritonOneShotBufferize
    : Pass<"libtriton-one-shot-bufferize", "mlir::ModuleOp"> {
  let summary = "LibTriton wrapper pass for one-shot bufferize";
  // clang-format off
  let description = [{
    Wraps upstream one-shot-bufferize behind a LibTriton pass entry so pipeline
    customization can evolve without changing Python pipeline strings.
  }];
  // clang-format on
  let options = [
    Option<"bufferizeFunctionBoundaries", "bufferize-function-boundaries",
           "bool", "false",
           "Bufferize function boundaries (forwarded to upstream pass).">,
    Option<
        "functionBoundaryTypeConversion", "function-boundary-type-conversion",
        "std::string", "\"infer-layout-map\"",
        "Layout map policy for function boundary conversion: "
        "infer-layout-map | identity-layout-map | fully-dynamic-layout-map.">
  ];
}
