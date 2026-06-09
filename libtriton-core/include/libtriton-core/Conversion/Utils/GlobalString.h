//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBTRITON_CORE_CONVERSION_UTILS_GLOBALSTRING_H_
#define LIBTRITON_CORE_CONVERSION_UTILS_GLOBALSTRING_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

namespace libtriton::conversion::utils {

/// Creates an LLVM global string constant with the ``__libtriton_constant_``
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

} // namespace libtriton::conversion::utils

#endif // LIBTRITON_CORE_CONVERSION_UTILS_GLOBALSTRING_H_
