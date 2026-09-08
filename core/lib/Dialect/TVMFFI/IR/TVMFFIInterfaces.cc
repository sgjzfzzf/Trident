//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TVMFFI/IR/TVMFFIInterfaces.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIInterfaces.cpp.inc"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h" // NOLINT(misc-include-cleaner)
#include <cstdint>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>

namespace trident::tvm_ffi {

ObjectResultOwnership CastOp::getObjectResultOwnership(uint32_t) {
  return ObjectResultOwnership::Borrowed;
}

ObjectOperandOwnership ObjectDecRefOp::getObjectOperandOwnership(uint32_t) {
  return ObjectOperandOwnership::Consumed;
}

ObjectOperandOwnership ObjectIncRefOp::getObjectOperandOwnership(uint32_t) {
  return ObjectOperandOwnership::Retained;
}

namespace detail {

struct FuncCallOwnershipModel final
    : ObjectOwnershipOpInterface::ExternalModel<FuncCallOwnershipModel,
                                                mlir::func::CallOp> {};

} // namespace detail

void registerTVMFFIObjectOwnershipExternalModels(
    mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *context, mlir::func::FuncDialect *) {
        mlir::func::CallOp::attachInterface<detail::FuncCallOwnershipModel>(
            *context);
      });
}

} // namespace trident::tvm_ffi
