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

include "trident/core/Dialect/TVMFFI/IR/TVMFFIBase.td"

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

// Trait for guard attrs whose lowering dereferences the value as a DLTensor*;
// the TVMFFIToLLVM pass emits a kTVMFFITensor pre-check for these.
// cppNamespace is cleared so the fully-qualified name below is used verbatim.
def TVMFFI_RequiresTensorGuardTrait
    : NativeAttrTrait<"::trident::tvm_ffi::RequiresTensorGuardTrait"> {
  let cppNamespace = "";
}

// Base for DLTensor*-dereferencing guard attrs.
class TVMFFI_TensorGuardAttr<string mnemonicStr>
    : TVMFFI_GuardAttr<mnemonicStr, [TVMFFI_RequiresTensorGuardTrait]>;

//===----------------------------------------------------------------------===//
// ConstantGuardAttr
//===----------------------------------------------------------------------===//

def TVMFFI_ConstantGuardAttr : TVMFFI_GuardAttr<"ConstantGuard"> {
  let description =
      [{Guards that a scalar variable equals an expected constant value.}];

  let parameters = (ins "int64_t" : $type_index, "int64_t" : $payload);
  let assemblyFormat =
      "`<` `type_index` `=` $type_index `,` `payload` `=` $payload `>`";
}

//===----------------------------------------------------------------------===//
// CUDADeviceGuardAttr
//===----------------------------------------------------------------------===//

def TVMFFI_CUDADeviceGuardAttr : TVMFFI_TensorGuardAttr<"CudaDeviceGuard"> {
  let description =
      [{Guards that a tensor variable resides on a specific CUDA device.}];

  let parameters = (ins "int64_t" : $device_type, "int64_t" : $device_index);
  let assemblyFormat = "`<` `device_type` `=` $device_type `,` `device_index` "
                       "`=` $device_index `>`";
}

//===----------------------------------------------------------------------===//
// DimensionGuardAttr
//===----------------------------------------------------------------------===//

def TVMFFI_DimensionGuardAttr : TVMFFI_TensorGuardAttr<"DimensionGuard"> {
  let description =
      [{Guards the number of dimensions(ndimension) of a tensor variable.}];

  let parameters = (ins "int64_t" : $expected);
  let assemblyFormat = "`<` `expected` `=` $expected `>`";
}

//===----------------------------------------------------------------------===//
// DTypeGuardAttr
//===----------------------------------------------------------------------===//

def TVMFFI_DTypeGuardAttr : TVMFFI_TensorGuardAttr<"DtypeGuard"> {
  let description =
      [{Guards the dtype of a tensor variable(e.g.torch.float32).}];

  let parameters =
      (ins "int64_t" : $code, "int64_t" : $bits, "int64_t" : $lanes);
  let assemblyFormat =
      "`<` `code` `=` $code `,` `bits` `=` $bits `,` `lanes` `=` $lanes `>`";
}

//===----------------------------------------------------------------------===//
// SizeGuardAttr
//===----------------------------------------------------------------------===//

def TVMFFI_SizeGuardAttr : TVMFFI_TensorGuardAttr<"SizeGuard"> {
  let description =
      [{Guards the size of a specific dimension of a tensor variable.}];

  let parameters = (ins "int64_t" : $index, "int64_t" : $expected);
  let assemblyFormat =
      "`<` `index` `=` $index `,` `expected` `=` $expected `>`";
}

//===----------------------------------------------------------------------===//
// StorageOffsetGuardAttr
//===----------------------------------------------------------------------===//

def TVMFFI_StorageOffsetGuardAttr : TVMFFI_TensorGuardAttr<"StorageOffsetGuard"> {
  let description = [{Guards the storage offset of a tensor variable.}];

  let parameters = (ins "int64_t" : $expected);
  let assemblyFormat = "`<` `expected` `=` $expected `>`";
}

//===----------------------------------------------------------------------===//
// StrideGuardAttr
//===----------------------------------------------------------------------===//

def TVMFFI_StrideGuardAttr : TVMFFI_TensorGuardAttr<"StrideGuard"> {
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
