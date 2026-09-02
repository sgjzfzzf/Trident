//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_DLPACK_IR_DLPACKOPS_TD_
#define TRIDENT_CORE_DIALECT_DLPACK_IR_DLPACKOPS_TD_

include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "trident/core/Dialect/DLPack/IR/DLPackBase.td"

class DLPack_Op<string mnemonic, list<Trait> traits = []>
    : Op<DLPack_Dialect, mnemonic, traits>;

class DLPack_TensorIndexedMetadataOp<string mnemonic>
    : DLPack_Op<mnemonic, [Pure]> {
  let arguments = (ins DLPack_DLTensorType:$tensor, I64:$index);
  let results = (outs I64:$result);
  let assemblyFormat = "$tensor `[` $index `]` attr-dict `:` type($tensor)";
}

def DLPack_TensorDataOp : DLPack_Op<"tensor.data", [Pure]> {
  let summary = "Read the data pointer from a DLPack tensor";
  let arguments = (ins DLPack_DLTensorType:$tensor);
  let results = (outs LLVM_AnyPointer:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `->` type($result)";
}

def DLPack_TensorDeviceOp : DLPack_Op<"tensor.device", [Pure]> {
  let summary = "Read DLPack device fields from a tensor";
  let arguments = (ins DLPack_DLTensorType:$tensor);
  let results = (outs I32:$device_type, I32:$device_index);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def DLPack_TensorDimOp : DLPack_Op<"tensor.dim", [Pure]> {
  let summary = "Read the rank of a DLPack tensor";
  let arguments = (ins DLPack_DLTensorType:$tensor);
  let results = (outs I64:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def DLPack_TensorDTypeOp : DLPack_Op<"tensor.dtype", [Pure]> {
  let summary = "Read DLPack dtype fields from a tensor";
  let arguments = (ins DLPack_DLTensorType:$tensor);
  let results = (outs I8:$code, I8:$bits, I16:$lanes);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def DLPack_TensorSizeOp
    : DLPack_TensorIndexedMetadataOp<"tensor.size"> {
  let summary = "Read a DLPack tensor dimension size";
}

def DLPack_TensorStorageOffsetOp
    : DLPack_Op<"tensor.storage_offset", [Pure]> {
  let summary = "Read a DLPack tensor storage offset";
  let arguments = (ins DLPack_DLTensorType:$tensor);
  let results = (outs I64:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def DLPack_TensorStrideOp
    : DLPack_TensorIndexedMetadataOp<"tensor.stride"> {
  let summary = "Read a DLPack tensor stride";
}

#endif // TRIDENT_CORE_DIALECT_DLPACK_IR_DLPACKOPS_TD_
