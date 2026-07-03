//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// trident-core-opt.cc - Standalone optimizer driver for TVMFFI and DLPack
// dialects.
//
// Registers core LLVM conversion pipeline passes and inserts dialects plus
// convert-to-LLVM interfaces needed by tests and development passes so
// trident-core-opt can parse and transform `.mlir` files exercising these
// dialects.

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "trident-core/Utils/Registration.h"

int main(int argc, char **argv) {
  trident::conversion::registerAllPasses();

  mlir::DialectRegistry registry;
  trident::conversion::registerAllDialects(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "TVM FFI modular optimizer driver\n", registry));
}
