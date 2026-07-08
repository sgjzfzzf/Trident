//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_UTILS_CHECK_H_
#define TRIDENT_CORE_CONVERSION_UTILS_CHECK_H_

#define TRIDENT_CHECK(Expr, OnFailure)                                         \
  ({                                                                           \
    auto trident_check_or = (Expr);                                            \
    if (mlir::failed(trident_check_or)) {                                      \
      OnFailure;                                                               \
    }                                                                          \
    *trident_check_or;                                                         \
  })

#define TRIDENT_CHECK_FAILURE(Expr) TRIDENT_CHECK(Expr, return mlir::failure())

#endif // TRIDENT_CORE_CONVERSION_UTILS_CHECK_H_
