//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// TVM FFI semantic types.
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFITYPES_H_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFITYPES_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
namespace mlir::TypeTrait {
template <typename ConcreteType>
class Object : public ::mlir::TypeTrait::TraitBase<ConcreteType, Object> {};
} // namespace mlir::TypeTrait

#define GET_TYPEDEF_CLASSES
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h.inc"

#endif
