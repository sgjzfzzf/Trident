//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "torch-mlir-c/Dialects.h"
#include "trident-core-c/Dialects.h"
#include "trident-core-c/Registration.h"

namespace nb = nanobind;

namespace {

void registerDialect(MlirDialectHandle handle, MlirContext context, bool load) {
  mlirDialectHandleRegisterDialect(handle, context);
  if (load) {
    mlirDialectHandleLoadDialect(handle, context);
  }
}

void registerTridentDialects(MlirContext context, bool load) {
  registerDialect(mlirGetDialectHandle__torchext__(), context, load);
  registerDialect(mlirGetDialectHandle__tvmffi__(), context, load);
  registerDialect(mlirGetDialectHandle__torch__(), context, load);
}

void registerAllDialects(MlirContext context, bool load) {
  if (load) {
    tridentCoreRegisterAllDialects(context);
    return;
  }
  registerTridentDialects(context, false);
}

void registerAllPasses() { tridentCoreRegisterAllPasses(); }

} // namespace

NB_MODULE(_tridentCore, m) {
  m.doc() = "trident-core python extension";

  m.def("register_all_dialects", &registerAllDialects, nb::arg("context"),
        nb::arg("load") = true);
  m.def("register_all_passes", &registerAllPasses);
}
