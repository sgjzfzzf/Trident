//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// TorchExtDialect.cc - TorchExt Dialect registration and initialization.
//
// This file implements the dialect `initialize()` method, which registers all
// ops and types defined in TorchExt.td / TorchExtTypes.td via
// ODS-generated .cpp.inc files.

#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TorchExt/IR/TorchExtOps.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TorchExt/IR/TorchExtTypes.h" // NOLINT(misc-include-cleaner)
#include <llvm/ADT/TypeSwitch.h> // NOLINT(misc-include-cleaner)
#include <llvm/Support/Casting.h>
#include <mlir/IR/Builders.h>              // NOLINT(misc-include-cleaner)
#include <mlir/IR/BuiltinTypes.h>          // NOLINT(misc-include-cleaner)
#include <mlir/IR/DialectImplementation.h> // NOLINT(misc-include-cleaner)
#include <mlir/Support/LLVM.h>             // NOLINT(misc-include-cleaner)
#include <torch-mlir/Dialect/Torch/IR/TorchTypes.h> // NOLINT(misc-include-cleaner)

using llvm::isa;

// The ODS-generated definitions below use declarations from these headers
// directly, so include-cleaner cannot infer their dependency from this file.
#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.cpp.inc"
#define GET_OP_CLASSES
#include "trident/core/Dialect/TorchExt/IR/TorchExt.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "trident/core/Dialect/TorchExt/IR/TorchExtTypes.cpp.inc"

namespace trident::torchext {

void TorchExtDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "trident/core/Dialect/TorchExt/IR/TorchExtTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "trident/core/Dialect/TorchExt/IR/TorchExt.cpp.inc"
      >();
}

} // namespace trident::torchext
