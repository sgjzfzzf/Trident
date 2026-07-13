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

#define GET_ATTRDEF_CLASSES
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIAttributes.h.inc"

#endif // TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIATTRIBUTES_H_
