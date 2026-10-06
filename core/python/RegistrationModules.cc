//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "dialects/torchext/TorchExtDialect.h"
#include "trident-c/core/Registration.h"
#include <mlir/Bindings/Python/IRCore.h>
#include <mlir/InitAllTranslations.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;

void registerAllDialects(
    mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
        context) {
  tridentCoreRegisterAllDialects(context.get()->get());
}

void registerAllPasses() { tridentCoreRegisterAllPasses(); }

NB_MODULE(_trident, m) {
  m.doc() = "trident-core python extension";

  // Translation registrations are global and must be installed before Python
  // creates an MLIRContext.  ExecutionEngine uses these interfaces when it
  // translates the lowered module to LLVM IR.
  mlir::registerToLLVMIRTranslation();

  m.def("register_all_dialects", &registerAllDialects, nb::arg("context"));
  m.def("register_all_passes", &registerAllPasses);
  mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::torchext::bindTorchExtTypes(m);
}
