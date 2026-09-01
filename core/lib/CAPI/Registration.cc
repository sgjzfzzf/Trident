//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-c/core/Registration.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/Torch/IR/TorchInterfaces.h"
#include "trident/core/Utils/Registration.h"
#include <mlir-c/IR.h>
#include <mlir/CAPI/IR.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>
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
  mlir::Type sourceType = unwrap(type);
  trident::torch::TorchToTVMFFITypeInterface const interface =
      mlir::dyn_cast<trident::torch::TorchToTVMFFITypeInterface>(sourceType);
  return wrap(
      interface ? interface.convertToTVMFFIType()
                : trident::tvm_ffi::AnyType::get(sourceType.getContext()));
}
