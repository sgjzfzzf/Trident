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

//===----------------------------------------------------------------------===//
// Triton kernel launch
//===----------------------------------------------------------------------===//

def TorchExt_TritonKernelLaunchOp : TorchExt_Op<"triton_kernel_launch", [
  AttrSizedOperandSegments, AllTypesMatch<[
    "gridSizeX", "gridSizeY", "gridSizeZ", "blockSizeX", "blockSizeY",
    "blockSizeZ"
  ]>
]> {
  let summary = "Launch a Torch kernel entry point";
  let description = [{
    Models launching a Torch kernel symbol with explicit grid/block sizes.
    Operands use Torch-MLIR types for staging initial TorchExt integration.

    Optionally supports cluster-level parallelism via `clusterSizeX`,
    `clusterSizeY`, and `clusterSizeZ` arguments. When present, the kernel
    is launched with the specified thread-block cluster size. This feature
    is exclusive to certain architectures (e.g. NVIDIA Hopper and later).
  }];

  let arguments = (ins SymbolRefAttr:$kernel,
                   Index:$gridSizeX,
                   Index:$gridSizeY,
                   Index:$gridSizeZ,
                   Index:$blockSizeX,
                   Index:$blockSizeY,
                   Index:$blockSizeZ,
                   Optional<Index>:$clusterSizeX,
                   Optional<Index>:$clusterSizeY,
                   Optional<Index>:$clusterSizeZ,
                   Optional<I32>:$dynamicSharedMemorySize,
                   Variadic<AnyTorchType>:$kernelOperands);
  let results = (outs);

  let extraClassDeclaration = [{
    /// Returns true if cluster size is specified.
    bool hasClusterSize();
  }];

  let assemblyFormat = [{
    $kernel
    (`clusters` `in` `(` $clusterSizeX^ `,` $clusterSizeY `,` $clusterSizeZ `)`)?
    `blocks` `in` `(` $gridSizeX `,` $gridSizeY `,` $gridSizeZ `)`
    `threads` `in` `(` $blockSizeX `,` $blockSizeY `,` $blockSizeZ `)`
    (`dynamic_shared_memory_size` $dynamicSharedMemorySize^)?
    (`args` `(` $kernelOperands^ `:` type($kernelOperands) `)`)?
    attr-dict
    `:` type($gridSizeX)
  }];
}

//===----------------------------------------------------------------------===//
// Stable list operations — map 1:1 to torch/csrc/stable/c/shim.h
//===----------------------------------------------------------------------===//

def TorchExt_ListDeleteListOp : TorchExt_Op<"aoti.ListDeleteList"> {
  let summary = "Delete a StableList via torch_delete_list()";
  // clang-format off
  let description = [{
    Deletes the underlying StableList referenced by the handle.

    Corresponds to:
      torch_delete_list(list_handle)
  }];
  // clang-format on

  let arguments = (ins AnyTorchListType : $list);
  let results = (outs);

  let assemblyFormat = "$list attr-dict `:` type($list)";
}

#endif // LIBTRITON_CORE_DIALECT_TORCHEXT_IR_TORCHEXTOPS_TD_
