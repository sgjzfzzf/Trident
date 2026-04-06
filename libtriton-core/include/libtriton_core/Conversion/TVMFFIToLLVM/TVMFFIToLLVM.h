#ifndef LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TVMFFITOLLVM_H_
#define LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TVMFFITOLLVM_H_

#include <memory>

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace libtriton::tvm_ffi {

void populateTVMFFIToLLVMTypeConversions(
    mlir::LLVMTypeConverter &typeConverter);
void populateTVMFFIToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns);

std::unique_ptr<mlir::Pass> createConvertTVMFFIToLLVMPass();
void registerConvertTVMFFIToLLVMPass();
void registerTVMFFIToLLVMPasses();
void registerConvertTVMFFIToLLVMInterface(mlir::DialectRegistry &registry);

} // namespace libtriton::tvm_ffi

#endif // LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TVMFFITOLLVM_H_
