//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TORCH_IR_TORCHINTERFACES_TD_
#define TRIDENT_CORE_DIALECT_TORCH_IR_TORCHINTERFACES_TD_

include "mlir/IR/OpBase.td"

def Torch_ToTVMFFITypeInterface
    : TypeInterface<"TorchToTVMFFITypeInterface"> {
  let cppNamespace = "::trident::torch";
  let description = [{
    Converts a frontend semantic type to its corresponding TVM FFI type.
  }];
  let methods = [
    InterfaceMethod<
      "Returns the corresponding semantic TVM FFI type.",
      "::mlir::Type", "getTVMFFIType", (ins)
    >
  ];
}

#endif // TRIDENT_CORE_DIALECT_TORCH_IR_TORCHINTERFACES_TD_
