//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_PYTHON_DIALECTS_TORCHEXT_TORCHEXTDIALECT_H
#define TRIDENT_CORE_PYTHON_DIALECTS_TORCHEXT_TORCHEXTDIALECT_H

#include <nanobind/nanobind.h>

namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::torchext {

void bindTorchExtTypes(nanobind::module_ &module);

} // namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::torchext

#endif // TRIDENT_CORE_PYTHON_DIALECTS_TORCHEXT_TORCHEXTDIALECT_H
