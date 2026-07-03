//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIATTRIBUTES_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIATTRIBUTES_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

include "trident-core/Dialect/TVMFFI/IR/TVMFFIBase.td"

// Base class for TVMFFI dialect attributes.
class TVMFFI_Attr<string name, list<Trait> traits = [],
                  string baseCppClass = "::mlir::Attribute">
    : AttrDef<TVMFFI_Dialect, name, traits, baseCppClass> {
  let mnemonic = name;
}

//===----------------------------------------------------------------------===//
// GuardAttribute — base class for guard-kind attributes
//===----------------------------------------------------------------------===//

class TVMFFI_GuardAttr<string mnemonicStr, list<Trait> traits = []>
    : TVMFFI_Attr<mnemonicStr, traits> {
  let summary = "A guard condition attribute for the TVM FFI tracing system";
}

//===----------------------------------------------------------------------===//
// CUDADeviceGuardAttr
//===----------------------------------------------------------------------===//

def TVMFFI_CUDADeviceGuardAttr : TVMFFI_GuardAttr<"CudaDeviceGuard"> {
  let description =
      [{Guards that a tensor variable resides on a specific CUDA device.}];

  let parameters = (ins "int64_t" : $device_type, "int64_t" : $device_index);
  let assemblyFormat = "`<` `device_type` `=` $device_type `,` `device_index` "
                       "`=` $device_index `>`";
}

//===----------------------------------------------------------------------===//
// DimensionGuardAttr
//===----------------------------------------------------------------------===//

def TVMFFI_DimensionGuardAttr : TVMFFI_GuardAttr<"DimensionGuard"> {
  let description =
      [{Guards the number of dimensions(ndimension) of a tensor variable.}];

  let parameters = (ins "int64_t" : $expected);
  let assemblyFormat = "`<` `expected` `=` $expected `>`";
}

//===----------------------------------------------------------------------===//
// DTypeGuardAttr
//===----------------------------------------------------------------------===//

def TVMFFI_DTypeGuardAttr : TVMFFI_GuardAttr<"DtypeGuard"> {
  let description =
      [{Guards the dtype of a tensor variable(e.g.torch.float32).}];

  let parameters = (ins);
  let assemblyFormat = "`<` `>`";
}

//===----------------------------------------------------------------------===//
// SizeGuardAttr
//===----------------------------------------------------------------------===//

def TVMFFI_SizeGuardAttr : TVMFFI_GuardAttr<"SizeGuard"> {
  let description =
      [{Guards the size of a specific dimension of a tensor variable.}];

  let parameters = (ins "int64_t" : $index, "int64_t" : $expected);
  let assemblyFormat =
      "`<` `index` `=` $index `,` `expected` `=` $expected `>`";
}

//===----------------------------------------------------------------------===//
// StorageOffsetGuardAttr
//===----------------------------------------------------------------------===//

def TVMFFI_StorageOffsetGuardAttr : TVMFFI_GuardAttr<"StorageOffsetGuard"> {
  let description = [{Guards the storage offset of a tensor variable.}];

  let parameters = (ins "int64_t" : $expected);
  let assemblyFormat = "`<` `expected` `=` $expected `>`";
}

//===----------------------------------------------------------------------===//
// StrideGuardAttr
//===----------------------------------------------------------------------===//

def TVMFFI_StrideGuardAttr : TVMFFI_GuardAttr<"StrideGuard"> {
  let description =
      [{Guards the stride of a specific dimension of a tensor variable.}];

  let parameters = (ins "int64_t" : $index, "int64_t" : $expected);
  let assemblyFormat =
      "`<` `index` `=` $index `,` `expected` `=` $expected `>`";
}

//===----------------------------------------------------------------------===//
// TensorTypeGuardAttr
//===----------------------------------------------------------------------===//

def TVMFFI_TensorTypeGuardAttr : TVMFFI_GuardAttr<"TensorTypeGuard"> {
  let description = [{Guards that a variable is of torch.Tensor type.}];

  let parameters = (ins);
  let assemblyFormat = "`<` `>`";
}

#endif // TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIATTRIBUTES_TD_
