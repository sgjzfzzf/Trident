#include "libtriton-core-c/DLPackTypes.h"
#include "libtriton-core-c/Dialects.h"
#include "libtriton-core-c/Registration.h"
#include "libtriton-core-c/TVMFFITypes.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "torch-mlir-c/Dialects.h"

namespace nb = nanobind;

NB_MODULE(_libtritonCore, m) {
  m.doc() = "libtriton-core python extension";

  m.def(
      "register_all_dialects",
      [](MlirContext context, bool load) {
        MlirDialectHandle dlpackHandle = mlirGetDialectHandle__dlpack__();
        MlirDialectHandle tvmffiHandle = mlirGetDialectHandle__tvm_ffi__();
        MlirDialectHandle torchHandle = mlirGetDialectHandle__torch__();

        mlirDialectHandleRegisterDialect(dlpackHandle, context);
        mlirDialectHandleRegisterDialect(tvmffiHandle, context);
        mlirDialectHandleRegisterDialect(torchHandle, context);
        if (load) {
          mlirDialectHandleLoadDialect(dlpackHandle, context);
          mlirDialectHandleLoadDialect(tvmffiHandle, context);
          mlirDialectHandleLoadDialect(torchHandle, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);
  m.def("register_all_passes", &libtritonCoreRegisterAllPasses);
}
