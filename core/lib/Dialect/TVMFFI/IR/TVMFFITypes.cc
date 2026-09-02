//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypeInterfaces.cpp.inc"
#include "trident/ffi/Exception.h"
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <tvm/ffi/object.h>

namespace trident::tvm_ffi {

int ExceptionType::getTypeIndex() {
  return tvm::ffi::TypeToRuntimeTypeIndex<trident::runtime::Exception>::v();
}

bool TVMFFIABIType::isObject() const {
  return static_cast<mlir::Type>(*this).hasTrait<mlir::TypeTrait::Object>();
}

bool TVMFFIABIType::classof(mlir::Type type) {
  return type && type.hasTrait<mlir::TypeTrait::TVMFFIABI>();
}

bool TVMFFIObjectType::classof(mlir::Type type) {
  return type && type.hasTrait<mlir::TypeTrait::Object>();
}

mlir::Type BoolType::getNativeType() const {
  return mlir::IntegerType::get(getContext(), 1);
}

mlir::Type FloatType::getNativeType() const {
  return mlir::Float64Type::get(getContext());
}

mlir::Type IntType::getNativeType() const {
  return mlir::IntegerType::get(getContext(), 64);
}

mlir::Type ArrayType::getNativeType() const {
  return ObjectType::get(getContext());
}

mlir::Type FunctionType::getNativeType() const {
  return ObjectType::get(getContext());
}

mlir::Type StrType::getNativeType() const {
  return ObjectType::get(getContext());
}

mlir::Type TensorType::getNativeType() const {
  return ObjectType::get(getContext());
}

mlir::LLVM::LLVMStructType
TVMFFIABIType::getLLVMType(mlir::MLIRContext *context) {
  mlir::IntegerType const i32Ty = mlir::IntegerType::get(context, 32);
  mlir::IntegerType const i64Ty = mlir::IntegerType::get(context, 64);
  return mlir::LLVM::LLVMStructType::getLiteral(context, {i32Ty, i32Ty, i64Ty});
}

mlir::Value IntType::build(mlir::OpBuilder &builder, mlir::Location loc,
                           int64_t value) {
  mlir::MLIRContext *context = builder.getContext();
  mlir::IntegerType const i32Ty = mlir::IntegerType::get(context, 32);
  mlir::IntegerType const i64Ty = mlir::IntegerType::get(context, 64);
  mlir::LLVM::LLVMPointerType const ptrTy =
      mlir::LLVM::LLVMPointerType::get(context);
  mlir::LLVM::LLVMStructType const anyTy = TVMFFIABIType::getLLVMType(context);
  mlir::Value const slot = mlir::LLVM::AllocaOp::create(
      builder, loc, ptrTy, anyTy,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
  mlir::Value any = mlir::LLVM::UndefOp::create(builder, loc, anyTy);
  any = mlir::LLVM::InsertValueOp::create(
      builder, loc, any,
      mlir::LLVM::ConstantOp::create(builder, loc, i32Ty,
                                     IntType::getTypeIndex()),
      llvm::ArrayRef<int64_t>{0});
  any = mlir::LLVM::InsertValueOp::create(
      builder, loc, any, mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, 0),
      llvm::ArrayRef<int64_t>{1});
  any = mlir::LLVM::InsertValueOp::create(
      builder, loc, any,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, value),
      llvm::ArrayRef<int64_t>{2});
  mlir::LLVM::StoreOp::create(builder, loc, any, slot);
  return slot;
}

mlir::Value IntType::build(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value value) {
  mlir::MLIRContext *context = builder.getContext();
  mlir::IntegerType const i64Ty = mlir::IntegerType::get(context, 64);
  mlir::LLVM::LLVMStructType const anyTy = TVMFFIABIType::getLLVMType(context);
  mlir::LLVM::LLVMPointerType const ptrTy =
      mlir::LLVM::LLVMPointerType::get(context);
  mlir::Value const slot = build(builder, loc, 0);
  mlir::LLVM::StoreOp::create(
      builder, loc, mlir::LLVM::SExtOp::create(builder, loc, i64Ty, value),
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, slot,
                                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2}));
  return slot;
}

mlir::Value DTypeType::build(mlir::OpBuilder &builder, mlir::Location loc,
                             int64_t code, int64_t bits, int64_t lanes) {
  int64_t const payload =
      (code & 0xff) | ((bits & 0xff) << 8) | ((lanes & 0xffff) << 16);
  mlir::MLIRContext *context = builder.getContext();
  mlir::Value const slot = IntType::build(builder, loc, payload);
  mlir::IntegerType const i32Ty = mlir::IntegerType::get(context, 32);
  mlir::LLVM::LLVMStructType const anyTy = TVMFFIABIType::getLLVMType(context);
  mlir::LLVM::StoreOp::create(
      builder, loc,
      mlir::LLVM::ConstantOp::create(builder, loc, i32Ty,
                                     DTypeType::getTypeIndex()),
      mlir::LLVM::GEPOp::create(
          builder, loc, mlir::LLVM::LLVMPointerType::get(context), anyTy, slot,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0}));
  return slot;
}

mlir::Value TVMFFIABIType::load(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value slot) {
  mlir::MLIRContext *context = builder.getContext();
  mlir::IntegerType const i64Ty = mlir::IntegerType::get(context, 64);
  mlir::LLVM::LLVMPointerType const ptrTy =
      mlir::LLVM::LLVMPointerType::get(context);
  mlir::LLVM::LLVMStructType const anyTy = TVMFFIABIType::getLLVMType(context);
  mlir::Value const payloadPtr =
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, slot,
                                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
  return mlir::LLVM::LoadOp::create(builder, loc, i64Ty, payloadPtr);
}

} // namespace trident::tvm_ffi
