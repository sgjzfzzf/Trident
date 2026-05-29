#ifndef LIBTRITON_CORE_DIALECT_TORCHEXT_IR_TORCHEXTOPS_TD_
#define LIBTRITON_CORE_DIALECT_TORCHEXT_IR_TORCHEXTOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Dialect/GPU/IR/GPUBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "torch-mlir/Dialect/Torch/IR/TorchTypes.td"

include "libtriton-core/Dialect/DLPack/IR/DLPackTypes.td"
include "libtriton-core/Dialect/TorchExt/IR/TorchExtBase.td"

class TorchExt_Op<string mnemonic, list<Trait> traits = []>
    : Op<TorchExt_Dialect, mnemonic, traits>;

def LaunchIndx : AnyTypeOf<[Index, I32, I64]>;

def TorchExt_KernelOperandType : AnyTypeOf<[
  AnySignlessInteger, AnyFloat, Index, AnyMemRef, AnyTensor,
  Torch_ValueTensorType
]>;

def TorchExt_TritonKernelLaunchOp : TorchExt_Op<"triton_kernel_launch", [
  GPU_AsyncOpInterface, AttrSizedOperandSegments, AllTypesMatch<[
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

    Optionally supports cluster-level parallelism via `clusterSizeX`,
    `clusterSizeY`, and `clusterSizeZ` arguments. When present, the kernel
    is launched with the specified thread-block cluster size. This feature
    is exclusive to certain architectures (e.g. NVIDIA Hopper and later).
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                   SymbolRefAttr:$kernel,
                   LaunchIndx:$gridSizeX,
                   LaunchIndx:$gridSizeY,
                   LaunchIndx:$gridSizeZ,
                   LaunchIndx:$blockSizeX,
                   LaunchIndx:$blockSizeY,
                   LaunchIndx:$blockSizeZ,
                   Optional<Index>:$clusterSizeX,
                   Optional<Index>:$clusterSizeY,
                   Optional<Index>:$clusterSizeZ,
                   Optional<I32>:$dynamicSharedMemorySize,
                   Variadic<TorchExt_KernelOperandType>:$kernelOperands,
                   Optional<AnyType>:$asyncObject);
  let results = (outs Optional<GPU_AsyncToken> : $asyncToken);

  let extraClassDeclaration = [{
    /// Returns true if cluster size is specified.
    bool hasClusterSize();
  }];

  let assemblyFormat = [{
    (`async` `[` $asyncDependencies^ `:` type($asyncDependencies) `]`)?
    (`<` $asyncObject^ `:` type($asyncObject) `>`)?
    $kernel
    (`clusters` `in` `(` $clusterSizeX^ `,` $clusterSizeY `,` $clusterSizeZ `)`)?
    `blocks` `in` `(` $gridSizeX `,` $gridSizeY `,` $gridSizeZ `)`
    `threads` `in` `(` $blockSizeX `,` $blockSizeY `,` $blockSizeZ `)`
    (`dynamic_shared_memory_size` $dynamicSharedMemorySize^)?
    (`args` `(` $kernelOperands^ `:` type($kernelOperands) `)`)?
    attr-dict
    (`->` type($asyncToken)^)?
    `:` type($gridSizeX)
  }];
}

def TorchExt_GetCurrentDeviceOp : TorchExt_Op<"get_current_device", [Pure]> {
  let summary = "Get current CUDA device index";
  let description = [{Returns the current CUDA device index from the runtime.}];

  let arguments = (ins);
  let results = (outs DLPack_DLDeviceType : $output);

  let assemblyFormat = "attr-dict `:` type($output)";
}

def TorchExt_GetCurrentStreamOp : TorchExt_Op<"get_current_stream", [Pure]> {
  let summary = "Get current CUDA stream async token";
  let description = [{
    Returns a gpu async token associated with the current CUDA stream for the
    provided device index. If the optional device attribute is omitted, lowering
    materializes -1.
  }];

  let arguments = (ins DefaultValuedAttr<I8Attr, "-1"> : $device);
  let results = (outs GPU_AsyncToken : $output);
  // clang-format off
  let assemblyFormat = [{
    `device` `=` $device attr-dict `:` type($output)
  }];
  // clang-format on
}

#endif // LIBTRITON_CORE_DIALECT_TORCHEXT_IR_TORCHEXTOPS_TD_
