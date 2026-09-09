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
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>
#include <torch-mlir/Dialect/Torch/IR/TorchDialect.h>
#include <torch-mlir/Dialect/Torch/IR/TorchTypes.h>

namespace trident::torch {
namespace detail {

template <typename ConcreteModel, typename SourceType, typename TargetType>
struct SimpleTypeModel
    : TorchToTVMFFITypeInterface::ExternalModel<ConcreteModel, SourceType> {
  mlir::Type getTVMFFIType(mlir::Type type) const {
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
struct UnionTypeModel final
    : TorchToTVMFFITypeInterface::ExternalModel<UnionTypeModel,
                                                mlir::torch::Torch::UnionType> {
  mlir::Type getTVMFFIType(mlir::Type type) const {
    mlir::torch::Torch::UnionType const unionType =
        mlir::cast<mlir::torch::Torch::UnionType>(type);
    llvm::SmallVector<mlir::Type> const convertedTypes = llvm::map_to_vector(
        unionType.getContainedTypes(),
        [&](mlir::Type containedType) -> mlir::Type {
          return llvm::TypeSwitch<mlir::Type, mlir::Type>(containedType)
              .Case<mlir::torch::Torch::AnyType>([&](mlir::Type) -> mlir::Type {
                return tvm_ffi::AnyType::get(type.getContext());
              })
              .Default([&](mlir::Type otherType) -> mlir::Type {
                TorchToTVMFFITypeInterface const interface =
                    mlir::dyn_cast<TorchToTVMFFITypeInterface>(otherType);
                return interface ? interface.getTVMFFIType()
                                 : tvm_ffi::AnyType::get(type.getContext());
              });
        });
    auto const anyType =
        llvm::find_if(convertedTypes, [](mlir::Type convertedType) -> bool {
          return mlir::isa<tvm_ffi::AnyType>(convertedType);
        });
    if (anyType != convertedTypes.end()) {
      return *anyType;
    }

    llvm::SmallVector<mlir::Type> flattenedTypes;
    auto const appendUnique = [&](mlir::Type member) {
      if (!llvm::is_contained(flattenedTypes, member)) {
        flattenedTypes.push_back(member);
      }
    };
    llvm::for_each(convertedTypes, [&](mlir::Type convertedType) {
      llvm::TypeSwitch<mlir::Type>(convertedType)
          .Case<tvm_ffi::UnionType>([&](tvm_ffi::UnionType convertedUnion) {
            llvm::for_each(convertedUnion.getTypes(), appendUnique);
          })
          .Default(appendUnique);
    });
    if (flattenedTypes.size() == 1) {
      return flattenedTypes.front();
    }
    if (flattenedTypes.empty()) {
      return tvm_ffi::AnyType::get(type.getContext());
    }
    return tvm_ffi::UnionType::get(type.getContext(), flattenedTypes);
  }
};
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
  mlir::Type getTVMFFIType(mlir::Type type) const {
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
    mlir::torch::Torch::UnionType::attachInterface<detail::UnionTypeModel>(
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
