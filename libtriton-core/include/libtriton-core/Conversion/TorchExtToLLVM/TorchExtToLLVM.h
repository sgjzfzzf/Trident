#ifndef LIBTRITON_CORE_CONVERSION_TORCHEXTTOLLVM_TORCHEXTTOLLVM_H_
#define LIBTRITON_CORE_CONVERSION_TORCHEXTTOLLVM_TORCHEXTTOLLVM_H_

#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace libtriton::torch_ext {

#define GEN_PASS_DECL_CONVERTTORCHEXTTOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTTORCHEXTTOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

void populateTorchExtToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns);

void registerConvertTorchExtToLLVMPass();
void registerConvertTorchExtToLLVMInterface(mlir::DialectRegistry &registry);

} // namespace libtriton::torch_ext

#endif // LIBTRITON_CORE_CONVERSION_TORCHEXTTOLLVM_TORCHEXTTOLLVM_H_
