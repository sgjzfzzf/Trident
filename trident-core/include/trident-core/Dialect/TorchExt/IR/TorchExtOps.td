//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TORCHEXT_IR_TORCHEXTOPS_TD_
#define TRIDENT_CORE_DIALECT_TORCHEXT_IR_TORCHEXTOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "torch-mlir/Dialect/Torch/IR/TorchTypes.td"

include "trident-core/Dialect/TorchExt/IR/TorchExtBase.td"

class TorchExt_Op<string mnemonic, list<Trait> traits = []>
    : Op<TorchExt_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Triton kernel launch
//===----------------------------------------------------------------------===//

def TorchExt_TridentKernelLaunchOp : TorchExt_Op<"trident_kernel_launch", [
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
// Object reference counting — inc-dec for Torch types backed by
// manually-managed resources.
//===----------------------------------------------------------------------===//

def TorchExt_ObjectIncRefOp : TorchExt_Op<"ObjectIncRef"> {
  let summary = "Increase the reference count of a Torch object";
  let description = [{
    Increases the reference count of the Torch object referenced by the value.
    This is used to transfer ownership of an object (e.g. when returning a
    value from a function, the caller receives an additional reference).

    Corresponds to:
      TVMFFIObjectIncRef(handle)
  }];

  let arguments = (ins AnyTorchType : $object);
  let results = (outs);

  let assemblyFormat = "$object attr-dict `:` type($object)";
}

def TorchExt_ObjectDecRefOp : TorchExt_Op<"ObjectDecRef"> {
  let summary = "Decrease the reference count of a Torch object";
  let description = [{
    Decreases the reference count of the Torch object referenced by the value.
    When the count reaches zero the underlying resource is released.

    Corresponds to:
      TVMFFIObjectDecRef(handle)

    This is the general-purpose op for any Torch type backed by a
    manually-managed resource (list, tensor, string, etc.).
  }];

  let arguments = (ins AnyTorchType : $object);
  let results = (outs);

  let assemblyFormat = "$object attr-dict `:` type($object)";
}

#endif // TRIDENT_CORE_DIALECT_TORCHEXT_IR_TORCHEXTOPS_TD_
