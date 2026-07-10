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
include "mlir/Interfaces/CastInterfaces.td"
include "torch-mlir/Dialect/Torch/IR/TorchTypes.td"

include "trident-core/Dialect/TorchExt/IR/TorchExtBase.td"

class TorchExt_Op<string mnemonic, list<Trait> traits = []>
    : Op<TorchExt_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Cast
//===----------------------------------------------------------------------===//

def TorchExt_CastOp
    : TorchExt_Op<"cast", [Pure, DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "Cast a Torch scalar (!torch.float or !torch.int) to a native "
                "MLIR scalar type";
  let description = [{
    Converts a `!torch.float` or `!torch.int` scalar value to a native MLIR
    type (f32, f64, i32, or i64).  This is necessary when passing scalar
    constants to external kernels (e.g. Triton) that expect a specific
    bit-width rather than the generic torch scalar representation.

    Example:
    ```
      %f32 = torchext.cast %f to f32 : !torch.float -> f32
      %i32 = torchext.cast %i to i32 : !torch.int -> i32
    ```
  }];

  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];
}

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
    Kernel operands may be Torch-MLIR tensor types or native scalar types
    (f32, f64, i32, i64) produced by `torchext.cast`.

    Optionally supports cluster-level parallelism via `clusterSizeX`,
    `clusterSizeY`, and `clusterSizeZ` arguments. When present, the kernel
    is launched with the specified thread-block cluster size. This feature
    is exclusive to certain architectures (e.g. NVIDIA Hopper and later).

}];

let arguments = (ins SymbolRefAttr:$kernel,
                   I64:$gridSizeX,
                   I64:$gridSizeY,
                   I64:$gridSizeZ,
                   I64:$blockSizeX,
                   I64:$blockSizeY,
                   I64:$blockSizeZ,
                   Optional<I64>:$clusterSizeX,
                   Optional<I64>:$clusterSizeY,
                   Optional<I64>:$clusterSizeZ,
                   Optional<I32>:$dynamicSharedMemorySize,
                   Variadic<AnyType>:$kernelOperands);
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
