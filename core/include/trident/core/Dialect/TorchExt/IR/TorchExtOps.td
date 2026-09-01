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

include "trident/core/Dialect/TorchExt/IR/TorchExtBase.td"
include "trident/core/Dialect/TorchExt/IR/TorchExtTypes.td"
include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.td"

class TorchExt_Op<string mnemonic, list<Trait> traits = []>
    : Op<TorchExt_Dialect, mnemonic, traits>;

def TorchExt_GetOperandType
    : AnyTypeOf<
          [Torch_BoolType, Torch_FloatType, Torch_IntType, TVMFFI_BoolType,
           TVMFFI_FloatType, TVMFFI_IntType, TVMFFI_TensorType],
          "Torch or TVM FFI value accepted by torchext.get">;

def TorchExt_GetResultType
    : AnyTypeOf<[AnyFloat, AnyInteger, LLVM_AnyPointer],
                "Native value produced by torchext.get">;

def TorchExt_KernelOperandType
    : AnyTypeOf<[
      AnyFloat, AnyInteger, AnyTorchType, TVMFFI_BoolType, TVMFFI_FloatType,
      TVMFFI_IntType, TVMFFI_TensorType
    ],
                "Operand accepted by a Trident kernel launch">;

def TorchExt_GetOp
    : TorchExt_Op<"get", [Pure]> {
  let summary = "Extract a native value from a Torch or TVM FFI value";
  let description = [{
    Extracts a `!torch.float`, `!torch.int`, or boolean scalar value to a
    native MLIR type. This is necessary when passing scalar constants to
    external kernels (e.g. Triton) that expect a specific bit-width rather
    than the generic Torch or TVM FFI representation.

    Example:
    ```
      %f32 = torchext.get %f to f32 : !torch.float -> f32
      %i32 = torchext.get %i to i32 : !torch.int -> i32
    ```
  }];

  let arguments = (ins TorchExt_GetOperandType:$operand);
  let results = (outs TorchExt_GetResultType:$result);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    static bool areGetCompatible(::mlir::TypeRange inputs,
                                 ::mlir::TypeRange outputs);
  }];

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];
}

def TorchExt_ConvertOp
    : TorchExt_Op<"convert", [Pure]> {
  let summary = "Convert a wrapper dtype to a Torch ScalarType integer";
  let description = [{
    Converts a `!torchext.dtype` wrapper value to `!torch.int` for calling
    native TorchMLIR Aten operations.  The operation is lowered through the
    TVM FFI DLPack dtype representation and the Trident runtime conversion
    helper.
  }];

  let arguments = (ins TorchExt_DTypeType:$operand);
  let results = (outs Torch_IntType:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];
}

def TorchExt_TridentKernelLaunchOp
    : TorchExt_Op<"trident_kernel_launch", [
  AttrSizedOperandSegments, AllTypesMatch<[
    "gridSizeX", "gridSizeY", "gridSizeZ", "blockSizeX", "blockSizeY",
    "blockSizeZ"
  ]>
]> {
  let summary = "Launch a Torch kernel entry point";
  let description = [{
    Models launching a Torch kernel symbol with explicit grid/block sizes.
    Kernel operands may be Torch-MLIR tensor types or native scalar types
    (f32, f64, i32, i64) produced by `torchext.get`.

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
                   Variadic<TorchExt_KernelOperandType>:$kernelOperands);
let results = (outs);

let extraClassDeclaration = [{
    /// Returns true if cluster size is specified.
    bool hasClusterSize();
  }];

  let assemblyFormat = [{
    $kernel
    (`clusters` `in` `(` $clusterSizeX^ `,` $clusterSizeY `,` $clusterSizeZ `)`)?
    `blocks` `in` `(` $gridSizeX `,` $gridSizeY `,` $gridSizeZ `)`
    `:` type($gridSizeX)
    `threads` `in` `(` $blockSizeX `,` $blockSizeY `,` $blockSizeZ `)`
    (`dynamic_shared_memory_size` $dynamicSharedMemorySize^)?
    (`args` `(` $kernelOperands^ `:` type($kernelOperands) `)`)?
    attr-dict
}];
}

#endif // TRIDENT_CORE_DIALECT_TORCHEXT_IR_TORCHEXTOPS_TD_
