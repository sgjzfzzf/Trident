//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

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
  let dependentDialects = ["mlir::cf::ControlFlowDialect"];
}

def ConvertTorchToTVMFFI : Pass<"convert-torch-to-tvm-ffi", "mlir::ModuleOp"> {
  let summary = "Expose Torch runtime calls and ownership in TVM FFI IR";
  let description = [{
    Rewrites Torch ATen dispatcher calls to tvm_ffi.FunctionGetGlobal /
    tvm_ffi.FunctionCall. Torch list and tuple
    values share !tvm_ffi.array, and object results carry explicit ownership.
  }];
  let dependentDialects = ["trident::tvm_ffi::TVMFFIDialect"];
}

def ConvertTVMFFIToLLVM : Pass<"convert-tvm-ffi-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower TVMFFI dialect operations to LLVM dialect";
  let description = [{Lower TVMFFI dialect operations to LLVM dialect.}];
  let dependentDialects =
      ["mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect"];
}

def DecomposeTVMFFI : Pass<"decompose-tvm-ffi", "mlir::ModuleOp"> {
  let summary = "Decompose composite TVMFFI operations";
  let description = [{Decompose TVMFFI array operations into function calls.}];
  let dependentDialects = ["trident::tvm_ffi::TVMFFIDialect"];
}

def TridentLoweringPipeline
    : Pass<"trident-lowering-pipeline", "mlir::ModuleOp"> {
  let summary = "Lower Torch dialect operations to LLVM dialect";
  let description = [{Lower Torch dialect operations to LLVM dialect.}];
  let dependentDialects = [
    "trident::torchext::TorchExtDialect", "mlir::LLVM::LLVMDialect",
    "mlir::torch::TorchConversion::TorchConversionDialect"
  ];
}
