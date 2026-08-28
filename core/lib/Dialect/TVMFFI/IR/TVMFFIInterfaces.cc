//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TVMFFI/IR/TVMFFIInterfaces.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIInterfaces.cpp.inc"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h" // NOLINT(misc-include-cleaner)
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>

namespace trident::tvm_ffi {

ObjectResultOwnership ArrayGetItemOp::getObjectResultOwnership(uint32_t) {
  return ObjectResultOwnership::Owned;
}

ObjectResultOwnership CastOp::getObjectResultOwnership(uint32_t) {
  return ObjectResultOwnership::Borrowed;
}

ObjectOperandOwnership
FunctionCallOp::getObjectOperandOwnership(uint32_t index) {
  return index == 0 ? ObjectOperandOwnership::Consumed
                    : ObjectOperandOwnership::Borrowed;
}

ObjectOperandOwnership ObjectDecRefOp::getObjectOperandOwnership(uint32_t) {
  return ObjectOperandOwnership::Consumed;
}

ObjectOperandOwnership ObjectIncRefOp::getObjectOperandOwnership(uint32_t) {
  return ObjectOperandOwnership::Retained;
}

namespace detail {

struct FuncCallOwnershipModel
    : ObjectOwnershipOpInterface::ExternalModel<FuncCallOwnershipModel,
                                                mlir::func::CallOp> {};

struct SCFIfOwnershipModel
    : ObjectOwnershipOpInterface::ExternalModel<SCFIfOwnershipModel,
                                                mlir::scf::IfOp> {};

} // namespace detail

void registerTVMFFIObjectOwnershipExternalModels(
    mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *context, mlir::func::FuncDialect *) {
        mlir::func::CallOp::attachInterface<detail::FuncCallOwnershipModel>(
            *context);
      });
  registry.addExtension(
      +[](mlir::MLIRContext *context, mlir::scf::SCFDialect *) {
        mlir::scf::IfOp::attachInterface<detail::SCFIfOwnershipModel>(*context);
      });
}

} // namespace trident::tvm_ffi
