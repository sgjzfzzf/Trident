//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def ConvertArithExtToScf
    : Pass<"convert-arith-ext-to-scf", "mlir::ModuleOp"> {
  let summary = "Lower ArithExt operations to nested SCF operations";
  let description = [{
    Lower arithext.and_then to nested scf.if operations.
  }];
  let dependentDialects = [
    "trident::arithext::ArithExtDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];
}

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

def ConvertTorchExtToTVMFFI
    : Pass<"convert-torchext-to-tvm-ffi", "mlir::ModuleOp"> {
  let summary = "Lower TorchExt conversion operations to TVM FFI operations";
  let description = [{Lower TorchExt conversion operations to TVM FFI operations.}];
  let dependentDialects = ["trident::tvm_ffi::TVMFFIDialect"];
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
  let dependentDialects = [
    "trident::tvm_ffi::TVMFFIDialect", "mlir::LLVM::LLVMDialect"
  ];
}

def ConvertTVMFFIToLLVM : Pass<"convert-tvm-ffi-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower TVMFFI dialect operations to LLVM dialect";
  let description = [{Lower TVMFFI dialect operations to LLVM dialect.}];
  let dependentDialects =
      ["mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect"];
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

def DecomposeTVMFFI : Pass<"decompose-tvm-ffi", "mlir::ModuleOp"> {
  let summary = "Decompose composite TVMFFI operations";
  let description = [{Decompose TVMFFI array operations into function calls.}];
  let dependentDialects = ["trident::tvm_ffi::TVMFFIDialect"];
}

def GeneralizeAtenOps : Pass<"generalize-aten-ops", "mlir::ModuleOp"> {
  let summary = "Represent concrete ATen operations as opaque Torch operators";
  let description = [{
    Rewrites concrete torch.aten operations to equivalent torch.operator
    operations so backend lowering does not invoke value-semantic ATen folders.
  }];
  let dependentDialects = ["mlir::torch::Torch::TorchDialect"];
}

def TridentLoweringPipeline
    : Pass<"trident-lowering-pipeline", "mlir::ModuleOp"> {
  let summary = "Lower Torch dialect operations to LLVM dialect";
  let description = [{Lower Torch dialect operations to LLVM dialect.}];
  let dependentDialects = [
    "trident::arithext::ArithExtDialect", "trident::torchext::TorchExtDialect",
    "mlir::arith::ArithDialect", "mlir::LLVM::LLVMDialect",
    "mlir::scf::SCFDialect",
    "mlir::torch::Torch::TorchDialect",
    "mlir::torch::TorchConversion::TorchConversionDialect"
  ];
}
