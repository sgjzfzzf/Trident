//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_UTILS_UNWRAP_H_
#define TRIDENT_CORE_CONVERSION_UTILS_UNWRAP_H_

#define TRIDENT_UNWRAP(Expr, OnFailure)                                        \
  ({                                                                           \
    auto trident_unwrap_or = (Expr);                                           \
    if (mlir::failed(trident_unwrap_or)) {                                     \
      OnFailure;                                                               \
    }                                                                          \
    *trident_unwrap_or;                                                        \
  })

#define TRIDENT_UNWRAP_FAILURE(Expr)                                           \
  TRIDENT_UNWRAP(Expr, return mlir::failure())

#endif // TRIDENT_CORE_CONVERSION_UTILS_UNWRAP_H_