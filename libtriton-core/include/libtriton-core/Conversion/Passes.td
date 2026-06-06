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
  let summary =
      "Lower TorchExt launch and runtime ops to GPU/LLVM-compatible form";
  // clang-format off
  let description = [{
    Rewrites torch_ext.triton_kernel_launch into gpu.launch_func while keeping
      async launch semantics, and lowers torch_ext.get_current_device and
      torch_ext.get_current_stream into LibTriton runtime C API calls.
  }];
  // clang-format on
  let dependentDialects = [
    "libtriton::torch_ext::TorchExtDialect", "mlir::func::FuncDialect",
    "mlir::gpu::GPUDialect", "mlir::LLVM::LLVMDialect"
  ];
}

def ConvertTorchToArith
    : Pass<"libtriton-convert-torch-to-arith", "mlir::ModuleOp"> {
  let summary = "Lower Torch dialect constant ops to Arith dialect";
  // clang-format off
  let description = [{
    Converts Torch dialect constant operations (torch.constant.bool,
    torch.constant.int, torch.constant.float) into the Arith dialect
    (arith.constant).
  }];
  // clang-format on
  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def ConvertTorchToAOTInductor
    : Pass<"convert-torch-to-aoti", "mlir::ModuleOp"> {
  let summary = "Lower Torch dialect operations to AOTInductor dialect";
  // clang-format off
  let description = [{
    Converts Torch dialect ops (e.g. torch.aten ops) into the AOTInductor
    dialect (aoti.torch_call_dispatcher etc.) for AOT-compiled model
    execution via the AOTInductor C shim interface.
  }];
  // clang-format on
  let dependentDialects = ["libtriton::aoti::AOTInductorDialect"];
}

def ConvertTorchToCf : Pass<"convert-torch-to-cf", "mlir::ModuleOp"> {
  let summary = "Lower Torch dialect runtime ops to ControlFlow dialect";
  let description = [{Lower Torch dialect runtime ops to ControlFlow dialect.}];
  let dependentDialects = ["mlir::cf::ControlFlowDialect"];
}

def ConvertTorchToLLVM : Pass<"convert-torch-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower Torch dialect operations to LLVM dialect";
  let description = [{Lower Torch dialect operations to LLVM dialect.}];
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}
