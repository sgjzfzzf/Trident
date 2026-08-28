//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// TVM FFI semantic types.
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFITYPES_H_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFITYPES_H_

#include "trident/core/Dialect/TVMFFI/IR/TVMFFIInterfaces.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypeInterfaces.h.inc"
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <torch-mlir/Dialect/Torch/IR/TorchTypes.h>

namespace mlir::TypeTrait {

template <typename ConcreteType>
class Object : public ::mlir::TypeTrait::TraitBase<ConcreteType, Object> {};

template <typename ConcreteType>
class TVMFFIABI : public ::mlir::TypeTrait::TraitBase<ConcreteType, TVMFFIABI> {
};

} // namespace mlir::TypeTrait

namespace trident::tvm_ffi {

class TVMFFIABIType : public ::mlir::Type {
public:
  using ::mlir::Type::Type;

  bool isObject() const;
  static int getTypeIndex();
  static ::mlir::LLVM::LLVMStructType getLLVMType(::mlir::MLIRContext *context);
  static ::mlir::Value load(::mlir::OpBuilder &builder, ::mlir::Location loc,
                            ::mlir::Value slot);
  static bool classof(::mlir::Type type);
};

class TVMFFIObjectType : public TVMFFIABIType {
public:
  using TVMFFIABIType::TVMFFIABIType;

  static bool classof(::mlir::Type type);
};

} // namespace trident::tvm_ffi

#define GET_TYPEDEF_CLASSES
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h.inc"

#endif
