include "mlir/Pass/PassBase.td"

def ConvertTorchConversionToLLVM
    : Pass<"convert-torch-conversion-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower TorchConversion dialect operations to LLVM dialect";
  let description =
      [{Lower TorchConversion dialect operations to LLVM dialect.}];
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::torch::TorchConversion::TorchConversionDialect"
  ];
}

def ConvertTorchExtToGPU : Pass<"convert-torchext-to-gpu", "mlir::ModuleOp"> {
  let summary = "Lower Torch Ext dialect operations to GPU dialect ";
      let description = [{Lower TorchExt dialect operations to GPU dialect.}];
  let dependentDialects = ["mlir::gpu::GPUDialect", "mlir::LLVM::LLVMDialect"];
}

def ConvertTorchExtToLLVM : Pass<"convert-torchext-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower TorchExt dialect operations to LLVM dialect";
  let description = [{Lower TorchExt dialect operations to LLVM dialect.}];
}

def ConvertTorchToCf : Pass<"convert-torch-to-cf", "mlir::ModuleOp"> {
  let summary = "Lower Torch dialect runtime ops to ControlFlow dialect";
  let description = [{Lower Torch dialect runtime ops to ControlFlow dialect.}];
  let dependentDialects = ["mlir::cf::ControlFlowDi alect "];
}

def ConvertTorchToLLVM : Pass<"convert-torch-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower Torch dialect constant ops to LLVM dialect";
  // clang-format off
  let description = [{
    Lowers Torch dialect constant operations (torch.constant.bool,
    torch.constant.int, torch.constant.float) directly to LLVM::ConstantOp,
    bypassing arith.constant and TorchConversion bridge ops.
  }];
  // clang-format on
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

def ConvertTVMFFIToLLVM : Pass<"convert-tvm-ffi-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower TVMFFI dialect operations to LLVM dialect";
  let description = [{Lower TVMFFI dialect operations to LLVM dialect.}];
  let dependentDialects =
      ["mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect"];
}

def FuncBackendTypeConversion
    : Pass<"func-backend-type-conversion", "mlir::ModuleOp"> {
  let summary =
      "Convert Torch dialect types to LLVM types along function boundaries";
  // clang-format off
  let description = [{
    Converts func::FuncOp signatures and func::ReturnOp / func::CallOp operands
    from Torch dialect types to LLVM / builtin types (e.g. !torch.bool -> i1,
    !torch.int -> i64, !torch.float -> f64, !torch.vtensor -> llvm.ptr).
  }];
  // clang-format on
}

def TorchToLLVMPipeline : Pass<"torch-to-llvm-pipeline", "mlir::ModuleOp"> {
  let summary = "Lower Torch dialect operations to LLVM dialect";
  let description = [{Lower Torch dialect operations to LLVM dialect.}];
  let dependentDialects = [
    "libtriton::torchext::TorchExtDialect", "mlir::LLVM::LLVMDialect",
    "mlir::torch::TorchConversion::TorchConversionDialect"
  ];
}
