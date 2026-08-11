//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIATTRIBUTES_H_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIATTRIBUTES_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"

#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"

namespace trident::tvm_ffi {
/// Marker for guard attrs whose lowering dereferences the value as a
/// DLTensor*; the TVMFFIToLLVM pass emits a kTVMFFITensor pre-check for
/// these so non-tensor values (e.g. None) fail the guard instead of
/// segfaulting.
template <typename ConcreteType>
struct RequiresTensorGuardTrait
    : public mlir::AttributeTrait::TraitBase<ConcreteType,
                                             RequiresTensorGuardTrait> {};
} // namespace trident::tvm_ffi

#define GET_ATTRDEF_CLASSES
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIAttributes.h.inc"

#endif // TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIATTRIBUTES_H_
