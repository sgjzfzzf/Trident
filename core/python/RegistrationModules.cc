//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-c/core/Registration.h"
#include <mlir-c/IR.h>
#include <mlir/Bindings/Python/NanobindAdaptors.h> // NOLINT(misc-include-cleaner)
#include <mlir/InitAllTranslations.h>
#include <nanobind/nanobind.h>
#include <nanobind/nb_defs.h>

namespace nb = nanobind;

namespace {

void registerAllDialects(MlirContext context) {
  tridentCoreRegisterAllDialects(context);
}

void registerAllPasses() { tridentCoreRegisterAllPasses(); }

MlirType convertTorchTypeToTVMFFIType(MlirType type) {
  return tridentCoreConvertTorchTypeToTVMFFIType(type);
}

} // namespace

NB_MODULE(_trident, m) {
  m.doc() = "trident-core python extension";

  // Translation registrations are global and must be installed before Python
  // creates an MLIRContext.  ExecutionEngine uses these interfaces when it
  // translates the lowered module to LLVM IR.
  mlir::registerToLLVMIRTranslation();

  m.def("register_all_dialects", &registerAllDialects, nb::arg("context"));
  m.def("register_all_passes", &registerAllPasses);
  m.def("_convert_torch_type_to_tvm_ffi_type", &convertTorchTypeToTVMFFIType,
        nb::arg("type"));
}
