#ifndef LIBTRITON_CORE_CONVERSION_DLPACKTOLLVM_DLPACKTOLLVM_H_
#define LIBTRITON_CORE_CONVERSION_DLPACKTOLLVM_DLPACKTOLLVM_H_

#include <cstdint>
#include <memory>

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace libtriton::dlpack {

#define GEN_PASS_DECL_CONVERTDLPACKTOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTDLPACKTOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

void populateDLPackToLLVMTypeConversions(
    mlir::LLVMTypeConverter &typeConverter);
void populateDLPackToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns);

void registerConvertDLPackToLLVMPass();
void registerConvertDLPackToLLVMInterface(mlir::DialectRegistry &registry);

} // namespace libtriton::dlpack

#endif // LIBTRITON_CORE_CONVERSION_DLPACKTOLLVM_DLPACKTOLLVM_H_
