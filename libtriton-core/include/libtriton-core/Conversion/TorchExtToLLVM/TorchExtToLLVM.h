#ifndef LIBTRITON_CORE_CONVERSION_TORCHEXTTOLLVM_TORCHEXTTOLLVM_H_
#define LIBTRITON_CORE_CONVERSION_TORCHEXTTOLLVM_TORCHEXTTOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace libtriton::torchext {

#define GEN_PASS_DECL_CONVERTTORCHEXTTOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTTORCHEXTTOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

void populateTorchExtToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns);

void registerConvertTorchExtToLLVMPass();
void registerConvertTorchExtToLLVMInterface(mlir::DialectRegistry &registry);

} // namespace libtriton::torchext

#endif // LIBTRITON_CORE_CONVERSION_TORCHEXTTOLLVM_TORCHEXTTOLLVM_H_
