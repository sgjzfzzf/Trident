#include <cstdint>

#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "torch-mlir-c/Dialects.h"
#include "torch-mlir-c/Registration.h"

namespace nb = nanobind;

NB_MODULE(_torchMlirCore, m) {
  torchMlirRegisterAllPasses();

  m.doc() = "libtriton-core torch registration extension";

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__torch__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);

  m.def("get_int64_max", []() { return INT64_MAX; });
  m.def("get_int64_min", []() { return INT64_MIN; });
}
