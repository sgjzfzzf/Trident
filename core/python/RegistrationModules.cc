//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "torch-mlir-c/Dialects.h"
#include "trident-c/core/Dialects.h"
#include "trident-c/core/Registration.h"

namespace nb = nanobind;

namespace {

void registerAllDialects(MlirContext context) {
  tridentCoreRegisterAllDialects(context);
}

void registerAllPasses() { tridentCoreRegisterAllPasses(); }

} // namespace

NB_MODULE(_trident, m) {
  m.doc() = "trident-core python extension";

  m.def("register_all_dialects", &registerAllDialects, nb::arg("context"));
  m.def("register_all_passes", &registerAllPasses);
}
