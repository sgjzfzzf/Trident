//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_UTILS_STRING_H_
#define TRIDENT_CORE_CONVERSION_UTILS_STRING_H_

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>

namespace trident::conversion::utils {

/// Allocates a null-terminated string on the stack and returns a pointer to
/// its first byte.
mlir::Value getString(mlir::OpBuilder &builder, mlir::Location loc,
                      llvm::StringRef content);

} // namespace trident::conversion::utils

#endif // TRIDENT_CORE_CONVERSION_UTILS_STRING_H_
