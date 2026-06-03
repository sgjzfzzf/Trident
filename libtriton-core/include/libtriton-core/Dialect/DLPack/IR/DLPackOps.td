#ifndef LIBTRITON_CORE_DIALECT_DLPACK_IR_DLPACKOPS_TD_
#define LIBTRITON_CORE_DIALECT_DLPACK_IR_DLPACKOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "libtriton-core/Dialect/DLPack/IR/DLPackBase.td"
include "libtriton-core/Dialect/DLPack/IR/DLPackTypes.td"

class DLPack_Op<string mnemonic, list<Trait> traits = []>
    : Op<DLPack_Dialect, mnemonic, traits>;

class DLPack_TensorOp<string mnemonic, list<Trait> traits = []>
    : DLPack_Op<mnemonic, traits> {
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let arguments = (ins DLPack_DLTensor : $input);
}

class DLPack_DeviceFieldOp<string mnemonic, list<Trait> traits = []>
    : DLPack_Op<mnemonic, traits> {
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let arguments = (ins DLPack_DLDeviceType : $input);
}

class DLPack_DTypeFieldOp<string mnemonic, list<Trait> traits = []>
    : DLPack_Op<mnemonic, traits> {
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let arguments = (ins DLPack_DLDataTypeType : $input);
}

class DLPack_IndexedOp<string mnemonic, list<Trait> traits = []>
    : Op<DLPack_Dialect, mnemonic, traits> {
  let assemblyFormat = "$input `[` $index `]` attr-dict `:` type($input) `,` "
                       "type($index) `->` type($output)";
  let arguments = (ins DLPack_DLTensor : $input, Index : $index);
}

def DLPack_ViewOp : DLPack_Op<"view", [Pure]> {
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let summary = "Convert a DLManagedTensor handle to a plain DLTensor handle";
  // clang-format off
  let description =
      [{Extracts the DLTensor payload from a DLPack DLManagedTensor handle.
        This op is intended as an interoperability scaffold and can be lowered
        during conversion to LLVM dialect.}];
  // clang-format on
  let arguments = (ins DLPack_DLManagedTensor : $input);
  let results = (outs DLPack_DLTensor : $output);
}

def DLPack_ToMemRefOp
    : DLPack_TensorOp<"to_memref", [MemoryEffects<[MemRead]>]> {
  let summary = "Convert a DLTensor handle to an MLIR memref value";
  // clang-format off
  let description =
      [{Bridges a DLPack DLTensor handle value to an MLIR memref-typed value.
        This op is intended as an interoperability scaffold and can be lowered
        during conversion to LLVM dialect.}];
  // clang-format on
  let results = (outs AnyMemRef : $output);
}

def DLPack_NDimOp : DLPack_TensorOp<"ndim", [Pure]> {
  let summary = "Get the rank of a DLTensor handle";
  let description = [{Extracts the ndim field from a DLPack DLTensor handle.}];
  let results = (outs I32 : $output);
}

def DLPack_ShapeOp : DLPack_IndexedOp<"shape", [MemoryEffects<[MemRead]>]> {
  let summary = "Get one shape element from a DLTensor handle";
  let description = [{Extracts one shape entry from a DLPack DLTensor handle.}];
  let results = (outs I64 : $output);
}

def DLPack_StridesOp : DLPack_IndexedOp<"strides", [MemoryEffects<[MemRead]>]> {
  let summary = "Get one stride element from a DLTensor handle";
  let description =
      [{Extracts one stride entry from a DLPack DLTensor handle.}];
  let results = (outs I64 : $output);
}

def DLPack_ByteOffsetOp : DLPack_TensorOp<"byte_offset", [Pure]> {
  let summary = "Get the byte offset of a DLTensor handle";
  let description =
      [{Extracts the byte_offset field from a DLPack DLTensor handle.}];
  let results = (outs I64 : $output);
}

def DLPack_DTypeOp : DLPack_TensorOp<"dtype", [Pure]> {
  let summary = "Get the dtype descriptor of a DLTensor handle";
  let description = [{Extracts the dtype field from a DLPack DLTensor handle.}];
  let results = (outs DLPack_DLDataTypeType : $output);
}

def DLPack_DTypeCodeOp : DLPack_DTypeFieldOp<"dtype_code", [Pure]> {
  let summary = "Get the dtype code of a DLPack dtype descriptor";
  let description = [{Extracts the code field from a DLPack DLDataType.}];
  let results = (outs I8 : $output);
}

def DLPack_DTypeBitsOp : DLPack_DTypeFieldOp<"dtype_bits", [Pure]> {
  let summary = "Get the dtype bits of a DLPack dtype descriptor";
  let description = [{Extracts the bits field from a DLPack DLDataType.}];
  let results = (outs I8 : $output);
}

def DLPack_DTypeLanesOp : DLPack_DTypeFieldOp<"dtype_lanes", [Pure]> {
  let summary = "Get the dtype lanes of a DLPack dtype descriptor";
  let description = [{Extracts the lanes field from a DLPack DLDataType.}];
  let results = (outs I16 : $output);
}

def DLPack_DeviceOp : DLPack_TensorOp<"device", [Pure]> {
  let summary = "Get the device descriptor of a DLTensor handle";
  let description =
      [{Extracts the device field from a DLPack DLTensor handle.}];
  let results = (outs DLPack_DLDeviceType : $output);
}

def DLPack_DeviceIdOp : DLPack_DeviceFieldOp<"device_id", [Pure]> {
  let summary = "Get the device id of a DLPack device descriptor";
  let description = [{Extracts the device_id field from a DLPack DLDevice.}];
  let results = (outs I32 : $output);
}

def DLPack_DeviceTypeOp : DLPack_DeviceFieldOp<"device_type", [Pure]> {
  let summary = "Get the device type of a DLPack device descriptor";
  let description = [{Extracts the device_type field from a DLPack DLDevice.}];
  let results = (outs I32 : $output);
}

#endif // LIBTRITON_CORE_DIALECT_DLPACK_IR_DLPACKOPS_TD_