#ifndef LIBTRITON_CORE_CONVERSION_TORCHCONVERSIONTOLLVM_TORCHCONVERSIONTOLLVM_H_
#define LIBTRITON_CORE_CONVERSION_TORCHCONVERSIONTOLLVM_TORCHCONVERSIONTOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace libtriton::torch {

#define GEN_PASS_DECL_CONVERTTORCHCONVERSIONTOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTTORCHCONVERSIONTOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

void populateTorchConversionToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns);

void registerConvertTorchConversionToLLVMPass();
void registerConvertTorchConversionToLLVMInterface(
    mlir::DialectRegistry &registry);

} // namespace libtriton::torch

#endif // LIBTRITON_CORE_CONVERSION_TORCHCONVERSIONTOLLVM_TORCHCONVERSIONTOLLVM_H_
