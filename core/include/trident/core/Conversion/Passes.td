//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def ConvertDLPackToLLVM : Pass<"convert-dlpack-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower DLPack tensor metadata operations to LLVM dialect";
  let description = [{Lower DLPack tensor metadata operations to LLVM dialect.}];
  let dependentDialects = [
    "trident::dlpack::DLPackDialect", "mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

def ConvertTorchExtToGPU : Pass<"convert-torchext-to-gpu", "mlir::ModuleOp"> {
  let summary = "Lower Torch Ext dialect operations to GPU dialect ";
  let description = [{Lower TorchExt dialect operations to GPU dialect.}];
  let dependentDialects = [
    "trident::dlpack::DLPackDialect", "mlir::gpu::GPUDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

def ConvertTorchToCf : Pass<"convert-torch-to-cf", "mlir::ModuleOp"> {
  let summary = "Lower Torch dialect runtime ops to ControlFlow dialect";
  let description = [{Lower Torch dialect runtime ops to ControlFlow dialect.}];
  let dependentDialects = ["mlir::cf::ControlFlowDialect"];
}

def ConvertTorchToScf : Pass<"trident-convert-torch-to-scf", "mlir::ModuleOp"> {
  let summary = "Lower Torch conditional control flow to SCF";
  let description = [{
    Lowers torch.prim.If and torch.prim.If.yield while preserving result types.
  }];
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::torch::TorchConversion::TorchConversionDialect",
    "trident::torchext::TorchExtDialect"
  ];
}

def ConvertTorchToTVMFFI : Pass<"convert-torch-to-tvm-ffi", "mlir::ModuleOp"> {
  let summary = "Expose Torch runtime calls and ownership in TVM FFI IR";
  let description = [{
    Rewrites Torch ATen dispatcher calls to tvm_ffi.FunctionGetGlobal /
    tvm_ffi.FunctionCall. Torch list and tuple
    values share !tvm_ffi.array, and object results carry explicit ownership.
  }];
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::torch::TorchConversion::TorchConversionDialect",
    "trident::tvm_ffi::TVMFFIDialect"
  ];
}

def ConvertTVMFFIToFunc : Pass<"convert-tvm-ffi-to-func", "mlir::ModuleOp"> {
  let summary = "Convert TVMFFI functions to Func operations";
  let description = [{Convert TVMFFI functions to Any ABI Func operations.}];
  let dependentDialects = [
    "mlir::arith::ArithDialect", "mlir::cf::ControlFlowDialect",
    "mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect",
    "mlir::scf::SCFDialect"
  ];
}

def ConvertTVMFFIToLLVM : Pass<"convert-tvm-ffi-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower TVMFFI dialect operations to LLVM dialect";
  let description = [{Lower TVMFFI dialect operations to LLVM dialect.}];
  let dependentDialects =
      ["mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect"];
}

def FinalizeTVMFFI : Pass<"finalize-tvm-ffi", "mlir::ModuleOp"> {
  let summary = "Finalize TVMFFI tensor metadata operations through DLPack";
  let description = [{
    Rewrite TVMFFI tensor metadata operations into TVMFFI object views and
    DLPack tensor metadata operations.
  }];
  let dependentDialects = [
    "trident::dlpack::DLPackDialect", "trident::tvm_ffi::TVMFFIDialect"
  ];
}

def TridentLoweringPipeline
    : Pass<"trident-lowering-pipeline", "mlir::ModuleOp"> {
  let summary = "Lower Torch dialect operations to LLVM dialect";
  let description = [{Lower Torch dialect operations to LLVM dialect.}];
  let dependentDialects = [
    "trident::torchext::TorchExtDialect",
    "mlir::arith::ArithDialect", "mlir::LLVM::LLVMDialect",
    "mlir::scf::SCFDialect",
    "mlir::torch::Torch::TorchDialect",
    "mlir::torch::TorchConversion::TorchConversionDialect"
  ];
}
