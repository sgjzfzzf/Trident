//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_PYTHON_DIALECTS_TORCH_TORCHDIALECT_H
#define TRIDENT_CORE_PYTHON_DIALECTS_TORCH_TORCHDIALECT_H

#include <nanobind/nanobind.h>

namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::torch {

void bindTorchTypes(nanobind::module_ &module);

} // namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::torch

#endif // TRIDENT_CORE_PYTHON_DIALECTS_TORCH_TORCHDIALECT_H
