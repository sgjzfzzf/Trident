#include "libtriton-core-c/Dialects.h"
#include "libtriton-core-c/Registration.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "torch-mlir-c/Dialects.h"

namespace nb = nanobind;

namespace {

void registerDialect(MlirDialectHandle handle, MlirContext context, bool load) {
  mlirDialectHandleRegisterDialect(handle, context);
  if (load) {
    mlirDialectHandleLoadDialect(handle, context);
  }
}

void registerLibTritonDialects(MlirContext context, bool load) {
  registerDialect(mlirGetDialectHandle__torchext__(), context, load);
  registerDialect(mlirGetDialectHandle__tvmffi__(), context, load);
  registerDialect(mlirGetDialectHandle__torch__(), context, load);
}

void registerAllDialects(MlirContext context, bool load) {
  if (load) {
    libtritonCoreRegisterAllDialects(context);
    return;
  }
  registerLibTritonDialects(context, false);
}

void registerAllPasses() { libtritonCoreRegisterAllPasses(); }

} // namespace

NB_MODULE(_libtritonCore, m) {
  m.doc() = "libtriton-core python extension";

  m.def("register_all_dialects", &registerAllDialects, nb::arg("context"),
        nb::arg("load") = true);
  m.def("register_all_passes", &registerAllPasses);
}
