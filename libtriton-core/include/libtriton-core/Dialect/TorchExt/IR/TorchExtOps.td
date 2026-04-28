#ifndef LIBTRITON_CORE_DIALECT_TORCHEXT_IR_TORCHEXTOPS_TD_
#define LIBTRITON_CORE_DIALECT_TORCHEXT_IR_TORCHEXTOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "torch-mlir/Dialect/Torch/IR/TorchTypes.td"

include "libtriton-core/Dialect/TorchExt/IR/TorchExtBase.td"

class TorchExt_Op<string mnemonic, list<Trait> traits = []>
    : Op<TorchExt_Dialect, mnemonic, traits>;

def LaunchIndx : AnyTypeOf<[Index, I32, I64]>;

def TorchExt_KernelOperandType : AnyTypeOf<[
  AnySignlessInteger, AnyFloat, Index, AnyMemRef, AnyTensor,
  Torch_ValueTensorType
]>;

def TorchExt_TritonKernelLaunchOp : TorchExt_Op<"triton_kernel_launch", [
  AttrSizedOperandSegments, AllTypesMatch<[
    "gridSizeX", "gridSizeY", "gridSizeZ", "blockSizeX", "blockSizeY",
    "blockSizeZ"
  ]>
]> {
  let summary = "Launch a Torch kernel entry point";
  let description = [{
    Models launching a Torch kernel symbol with explicit grid/block sizes.
    In addition to regular scalar and memref launch arguments, this op also
    supports builtin tensor and torch vtensor operands for staging initial
    TorchExt integration.
  }];

  let arguments = (ins SymbolRefAttr:$kernel,
                   LaunchIndx:$gridSizeX,
                   LaunchIndx:$gridSizeY,
                   LaunchIndx:$gridSizeZ,
                   LaunchIndx:$blockSizeX,
                   LaunchIndx:$blockSizeY,
                   LaunchIndx:$blockSizeZ,
                   Optional<I32>:$dynamicSharedMemorySize,
                   Variadic<TorchExt_KernelOperandType>:$kernelOperands,
                   Optional<AnyType>:$asyncObject);
  let results = (outs);

  let assemblyFormat = [{
    $kernel
    (`<` $asyncObject^ `:` type($asyncObject) `>`)?
    `blocks` `in` `(` $gridSizeX `,` $gridSizeY `,` $gridSizeZ `)`
    `threads` `in` `(` $blockSizeX `,` $blockSizeY `,` $blockSizeZ `)`
    (`dynamic_shared_memory_size` $dynamicSharedMemorySize^)?
    (`args` `(` $kernelOperands^ `:` type($kernelOperands) `)`)?
    attr-dict
    `:` type($gridSizeX)
  }];
}

def TorchExt_GetCurrentDeviceOp : TorchExt_Op<"get_current_device", [Pure]> {
  let summary = "Get current CUDA device index";
  let description = [{Returns the current CUDA device index from the runtime.}];

  let arguments = (ins);
  let results = (outs I8 : $output);

  let assemblyFormat = "attr-dict `:` type($output)";
}

def TorchExt_GetCurrentStreamOp : TorchExt_Op<"get_current_stream", [Pure]> {
  let summary = "Get current CUDA stream pointer";
  let description = [{
    Returns the current CUDA stream pointer for the provided device index.
    If the optional device operand is omitted, lowering materializes -1.
  }];

  let arguments = (ins Optional<I8> : $device);
  let results = (outs LLVM_AnyPointer : $output);
  // clang-format off
  let assemblyFormat = [{
    (`device` $device^ `:` type($device))?
    attr-dict
    `:` type($output)
  }];
  // clang-format on
}

#endif // LIBTRITON_CORE_DIALECT_TORCHEXT_IR_TORCHEXTOPS_TD_
