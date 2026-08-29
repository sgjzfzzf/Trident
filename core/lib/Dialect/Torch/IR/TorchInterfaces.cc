//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/Torch/IR/TorchInterfaces.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/Torch/IR/TorchTypeInterfaces.cpp.inc"
#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtTypes.h"
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <torch-mlir/Dialect/Torch/IR/TorchDialect.h>
#include <torch-mlir/Dialect/Torch/IR/TorchTypes.h>

namespace trident::torch {
namespace detail {

template <typename ConcreteModel, typename SourceType, typename TargetType>
struct SimpleTypeModel
    : TorchToTVMFFITypeInterface::ExternalModel<ConcreteModel, SourceType> {
  mlir::Type convertToTVMFFIType(mlir::Type type) const {
    return TargetType::get(type.getContext());
  }
};

struct AnyTypeModel final
    : SimpleTypeModel<AnyTypeModel, mlir::torch::Torch::AnyType,
                      tvm_ffi::AnyType> {};
struct BoolTypeModel final
    : SimpleTypeModel<BoolTypeModel, mlir::torch::Torch::BoolType,
                      tvm_ffi::BoolType> {};
struct DeviceTypeModel final
    : SimpleTypeModel<DeviceTypeModel, mlir::torch::Torch::DeviceType,
                      tvm_ffi::DeviceType> {};
struct DTypeTypeModel final
    : SimpleTypeModel<DTypeTypeModel, torchext::DTypeType, tvm_ffi::DTypeType> {
};
struct FloatTypeModel final
    : SimpleTypeModel<FloatTypeModel, mlir::torch::Torch::FloatType,
                      tvm_ffi::FloatType> {};
struct IntTypeModel final
    : SimpleTypeModel<IntTypeModel, mlir::torch::Torch::IntType,
                      tvm_ffi::IntType> {};
struct ListTypeModel final
    : SimpleTypeModel<ListTypeModel, mlir::torch::Torch::ListType,
                      tvm_ffi::ArrayType> {};
struct TupleTypeModel final
    : SimpleTypeModel<TupleTypeModel, mlir::torch::Torch::TupleType,
                      tvm_ffi::ArrayType> {};
struct NoneTypeModel final
    : SimpleTypeModel<NoneTypeModel, mlir::torch::Torch::NoneType,
                      tvm_ffi::NoneType> {};
struct NonValueTensorTypeModel final
    : SimpleTypeModel<NonValueTensorTypeModel,
                      mlir::torch::Torch::NonValueTensorType,
                      tvm_ffi::TensorType> {};
struct ValueTensorTypeModel final
    : SimpleTypeModel<ValueTensorTypeModel, mlir::torch::Torch::ValueTensorType,
                      tvm_ffi::TensorType> {};

struct StringTypeModel final
    : TorchToTVMFFITypeInterface::ExternalModel<
          StringTypeModel, mlir::torch::Torch::StringType> {
  mlir::Type convertToTVMFFIType(mlir::Type type) const {
    return tvm_ffi::UnionType::get(
        type.getContext(), {tvm_ffi::RawStrType::get(type.getContext()),
                            tvm_ffi::SmallStrType::get(type.getContext()),
                            tvm_ffi::StrType::get(type.getContext())});
  }
};

} // namespace detail

void registerTorchToTVMFFITypeInterfaces(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *context,
                            mlir::torch::Torch::TorchDialect *) {
    mlir::torch::Torch::AnyType::attachInterface<detail::AnyTypeModel>(
        *context);
    mlir::torch::Torch::BoolType::attachInterface<detail::BoolTypeModel>(
        *context);
    mlir::torch::Torch::DeviceType::attachInterface<detail::DeviceTypeModel>(
        *context);
    mlir::torch::Torch::FloatType::attachInterface<detail::FloatTypeModel>(
        *context);
    mlir::torch::Torch::IntType::attachInterface<detail::IntTypeModel>(
        *context);
    mlir::torch::Torch::ListType::attachInterface<detail::ListTypeModel>(
        *context);
    mlir::torch::Torch::TupleType::attachInterface<detail::TupleTypeModel>(
        *context);
    mlir::torch::Torch::NoneType::attachInterface<detail::NoneTypeModel>(
        *context);
    mlir::torch::Torch::NonValueTensorType::attachInterface<
        detail::NonValueTensorTypeModel>(*context);
    mlir::torch::Torch::ValueTensorType::attachInterface<
        detail::ValueTensorTypeModel>(*context);
    mlir::torch::Torch::StringType::attachInterface<detail::StringTypeModel>(
        *context);
  });
  registry.addExtension(
      +[](mlir::MLIRContext *context, torchext::TorchExtDialect *) {
        torchext::DTypeType::attachInterface<detail::DTypeTypeModel>(*context);
      });
}

} // namespace trident::torch
