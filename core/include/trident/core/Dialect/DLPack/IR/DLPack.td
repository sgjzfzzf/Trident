//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_DLPACK_IR_DLPACK_TD_
#define TRIDENT_CORE_DIALECT_DLPACK_IR_DLPACK_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "trident/core/Dialect/DLPack/IR/DLPackBase.td"

def DLPack_DLDataType
    : TypeDef<DLPack_Dialect, "DLDataType", [],
              "::mlir::Type"> {
  let mnemonic = "dl_data_type";
  let extraClassDeclaration = [{
    static ::mlir::LLVM::LLVMStructType
    getLLVMType(::mlir::MLIRContext *context);
  }];
}

def DLPack_DLDeviceType
    : TypeDef<DLPack_Dialect, "DLDevice", [],
              "::mlir::Type"> {
  let mnemonic = "dl_device";
  let extraClassDeclaration = [{
    static ::mlir::LLVM::LLVMStructType
    getLLVMType(::mlir::MLIRContext *context);
  }];
}

def DLPack_DLTensorType
    : TypeDef<DLPack_Dialect, "DLTensor", [],
              "::mlir::Type"> {
  let mnemonic = "dl_tensor";
  let extraClassDeclaration = [{
    static ::mlir::LLVM::LLVMStructType
    getLLVMType(::mlir::MLIRContext *context);
  }];
}

include "trident/core/Dialect/DLPack/IR/DLPackOps.td"

#endif // TRIDENT_CORE_DIALECT_DLPACK_IR_DLPACK_TD_
