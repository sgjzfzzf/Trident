#include <stdio.h>

#include "libtriton-core-c/Registration.h"
#include "libtriton-core-c/TVMFFITypes.h"
#include "mlir-c/IR.h"

int main(void) {
  MlirContext context = mlirContextCreate();
  libtritonCoreRegisterAllDialects(context);

  MlirType any = libtritonCoreTVMFFIAnyTypeGet(context);
  int ok = libtritonCoreTypeIsATVMFFIAnyType(any);

  mlirContextDestroy(context);
  if (!ok) {
    fprintf(stderr, "expected TVMFFI Any type\n");
    return 1;
  }
  return 0;
}
