#include <stdio.h>

#include "libtriton-core-c/Dialects.h"
#include "libtriton-core-c/Registration.h"
#include "mlir-c/IR.h"

int main(void) {
  MlirContext context = mlirContextCreate();
  libtritonCoreRegisterAllDialects(context);

  MlirDialectHandle handle = mlirGetDialectHandle__torch_ext__();
  MlirDialect dialect = mlirDialectHandleLoadDialect(handle, context);
  int ok = !mlirDialectIsNull(dialect);

  mlirContextDestroy(context);
  if (!ok) {
    fprintf(stderr, "expected TorchExt dialect\n");
    return 1;
  }
  return 0;
}
