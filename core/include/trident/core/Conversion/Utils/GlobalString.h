//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_UTILS_GLOBALSTRING_H_
#define TRIDENT_CORE_CONVERSION_UTILS_GLOBALSTRING_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

namespace trident::conversion::utils {

/// Creates an LLVM global string constant with the ``__trident_constant_``
/// prefix and returns a pointer to it.  The content is null-terminated so that
/// C code can safely treat the pointer as a ``const char*``.
///
/// Skips creation if an identically-named global already exists.
///
/// \param name    Short label embedded in the symbol name for disambiguation
///                (e.g. ``"op"``, ``"overload"``, ``"kind"``, ``"msg"``).
/// \param content The string content (without null terminator — it is added
///                automatically).
mlir::Value getOrCreateGlobalString(mlir::OpBuilder &builder,
                                    mlir::Location loc, mlir::ModuleOp moduleOp,
                                    llvm::StringRef name,
                                    llvm::StringRef content);

} // namespace trident::conversion::utils

#endif // TRIDENT_CORE_CONVERSION_UTILS_GLOBALSTRING_H_
