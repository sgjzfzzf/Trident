//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_TORCHTOLLVM_TORCHTOLLVM_H_
#define TRIDENT_CORE_CONVERSION_TORCHTOLLVM_TORCHTOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "llvm/ADT/MapVector.h"

namespace trident::torch {

#define GEN_PASS_DECL_CONVERTTORCHTOLLVM
#include "trident/core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTTORCHTOLLVM
#include "trident/core/Conversion/Passes.h.inc"

/// True for Torch types whose values are backed by manually-managed FFI
/// objects that need reference counting (tensor / list / tuple / optional).
inline bool isRefCountedObjectType(mlir::Type type) {
  return llvm::isa<mlir::torch::Torch::ValueTensorType,
                   mlir::torch::Torch::NonValueTensorType,
                   mlir::torch::Torch::ListType, mlir::torch::Torch::TupleType,
                   mlir::torch::Torch::OptionalType>(type);
}

/// Reference-count table populated during Torch->LLVM conversion.
///
/// Keys are the converted LLVM object values (TVMFFIAny) produced by the
/// conversion patterns (Aten dispatcher, PrimListConstruct, ...); the value
/// is the net reference count accumulated during conversion (+1 per escape,
/// -1 per in-scope last use).  After conversion, this table drives the
/// insertion of TVMFFIObjectIncRef/DecRef calls for escaping / dead objects;
/// the running net counts also feed the debug output and can later be used
/// to pair-eliminate or merge adjacent Inc/Dec calls.
using RefCountTable = llvm::MapVector<mlir::Value, int>;

void populateTorchToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, RefCountTable &refCountTable);

void populateTorchToLLVMConstantConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns);

void populateTorchToLLVMLiteralConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns);

void populateTorchToLLVMAtenConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, RefCountTable &refCountTable);

void populateTorchToLLVMPrimConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, RefCountTable &refCountTable);

void registerConvertTorchToLLVMPass();
void registerConvertTorchToLLVMInterface(mlir::DialectRegistry &registry);

} // namespace trident::torch

#endif // TRIDENT_CORE_CONVERSION_TORCHTOLLVM_TORCHTOLLVM_H_
