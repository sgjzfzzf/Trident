#include <stdio.h>

#include "libtriton-core-c/DLPackTypes.h"
#include "libtriton-core-c/Registration.h"
#include "mlir-c/IR.h"

int main(void) {
  MlirContext context = mlirContextCreate();
  libtritonCoreRegisterAllDialects(context);

  MlirType dlTensor = libtritonCoreDLPackDLTensorTypeGet(context);
  int ok = libtritonCoreTypeIsADLPackDLTensorType(dlTensor);

  mlirContextDestroy(context);
  if (!ok) {
    fprintf(stderr, "expected DLPack DLTensor type\n");
    return 1;
  }
  return 0;
}
