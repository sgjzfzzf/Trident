//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIINTERFACES_H_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIINTERFACES_H_

#include <cstdint>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/OpDefinition.h>

namespace trident::tvm_ffi {

enum class ObjectOperandOwnership : std::uint8_t {
  Borrowed,
  Consumed,
  Retained,
};

enum class ObjectResultOwnership : std::uint8_t {
  Borrowed,
  Owned,
};

void registerTVMFFIObjectOwnershipExternalModels(
    mlir::DialectRegistry &registry);

} // namespace trident::tvm_ffi

#include "trident/core/Dialect/TVMFFI/IR/TVMFFIAttrInterfaces.h.inc"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIInterfaces.h.inc"

#endif // TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIINTERFACES_H_
