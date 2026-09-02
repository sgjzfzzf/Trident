//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_DLPACK_IR_DLPACKBASE_TD_
#define TRIDENT_CORE_DIALECT_DLPACK_IR_DLPACKBASE_TD_

include "mlir/IR/DialectBase.td"

def DLPack_Dialect : Dialect {
  let name = "dlpack";
  let cppNamespace = "::trident::dlpack";
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
  let useDefaultTypePrinterParser = 1;
  let summary = "DLPack dialect";
}

#endif // TRIDENT_CORE_DIALECT_DLPACK_IR_DLPACKBASE_TD_
