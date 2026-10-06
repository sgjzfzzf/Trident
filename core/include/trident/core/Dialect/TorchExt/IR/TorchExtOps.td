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
include "mlir/Interfaces/InferTypeOpInterface.td"

include "trident/core/Dialect/TorchExt/IR/TorchExtBase.td"
include "trident/core/Dialect/TorchExt/IR/TorchExtAttrs.td"
include "trident/core/Dialect/TorchExt/IR/TorchExtTypes.td"
include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.td"

class TorchExt_Op<string mnemonic, list<Trait> traits = []>
    : Op<TorchExt_Dialect, mnemonic, traits>;

def TorchExt_KernelOperandType
    : AnyTypeOf<[AnyTVMFFISemanticType, I1, I64, F64],
                "TVM FFI value or native scalar accepted by a Triton launch">;

def TorchExt_TritonKernelLaunchOp
    : TorchExt_Op<"trident_kernel_launch", [
  AttrSizedOperandSegments, AllTypesMatch<[
    "gridSizeX", "gridSizeY", "gridSizeZ", "blockSizeX", "blockSizeY",
    "blockSizeZ"
  ]>
]> {
  let summary = "Launch a Torch kernel entry point";
  let description = [{
    Models launching a Torch kernel symbol with explicit grid/block sizes.
    Operands use TVM FFI semantic values or native scalars. Each operand's
    specialization attribute defines its runtime precondition. Constant
    specializations are guard-only, while variable specializations are passed
    to the compiled Triton kernel.

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
                   Variadic<TorchExt_KernelOperandType>:$kernelOperands,
                   TorchExt_SpecializationArrayAttr:$specializations);
let results = (outs);
let hasVerifier = 1;

let extraClassDeclaration = [{
    /// Returns true if cluster size is specified.
    bool hasClusterSize();
    static ::mlir::ParseResult parseKernelArguments(
        ::mlir::OpAsmParser &parser,
        ::llvm::SmallVectorImpl<::mlir::OpAsmParser::UnresolvedOperand> &operands,
        ::llvm::SmallVectorImpl<::mlir::Type> &types,
        ::mlir::ArrayAttr &specializations);
    static void printKernelArguments(
        ::mlir::OpAsmPrinter &printer,
        ::mlir::Operation *op [[maybe_unused]],
        ::mlir::OperandRange operands, ::mlir::TypeRange types,
        ::mlir::ArrayAttr specializations);
  }];

  let assemblyFormat = [{
    $kernel
    (`clusters` `in` `(` $clusterSizeX^ `,` $clusterSizeY `,` $clusterSizeZ `)`)?
    `blocks` `in` `(` $gridSizeX `,` $gridSizeY `,` $gridSizeZ `)`
    `:` type($gridSizeX)
    `threads` `in` `(` $blockSizeX `,` $blockSizeY `,` $blockSizeZ `)`
    (`dynamic_shared_memory_size` $dynamicSharedMemorySize^)?
    `args` `(` custom<KernelArguments>($kernelOperands, type($kernelOperands), $specializations) `)`
    prop-dict attr-dict
}];
}

#endif // TRIDENT_CORE_DIALECT_TORCHEXT_IR_TORCHEXTOPS_TD_
