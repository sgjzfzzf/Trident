//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-c/core/Registration.h"
#include "trident/core/Conversion/TorchToTVMFFI/TorchToTVMFFI.h"
#include "trident/core/Utils/Registration.h"
#include <mlir-c/IR.h>
#include <mlir/CAPI/IR.h>
#include <mlir/IR/DialectRegistry.h>
#include <torch-mlir-c/Registration.h>

void tridentCoreRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  trident::conversion::registerAllDialects(registry);
  unwrap(context)->appendDialectRegistry(registry);
  torchMlirRegisterAllDialects(context);
  unwrap(context)->loadAllAvailableDialects();
}

void tridentCoreRegisterAllPasses(void) {
  trident::conversion::registerAllPasses();
}

MlirType tridentCoreConvertTorchTypeToTVMFFIType(MlirType type) {
  return wrap(trident::conversion::convertTorchTypeToTVMFFIType(unwrap(type)));
}
