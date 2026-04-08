#ifndef LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TVMFFITOLLVM_H_
#define LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TVMFFITOLLVM_H_

#include <memory>

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace libtriton::tvm_ffi {

#define GEN_PASS_DECL_CONVERTTVMFFITOLLVM
#include "libtriton_core/Conversion/TVMFFIToLLVM/Passes.h.inc"

#define GEN_PASS_DECL_EMITTVMFFIINTERFACE
#include "libtriton_core/Conversion/TVMFFIToLLVM/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTTVMFFITOLLVM
#include "libtriton_core/Conversion/TVMFFIToLLVM/Passes.h.inc"

#define GEN_PASS_REGISTRATION_EMITTVMFFIINTERFACE
#include "libtriton_core/Conversion/TVMFFIToLLVM/Passes.h.inc"

void populateTVMFFIToLLVMTypeConversions(
    mlir::LLVMTypeConverter &typeConverter);
void populateTVMFFIToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns);

void registerConvertTVMFFIToLLVMPass();
void registerEmitTVMFFIInterfacePass();
void registerTVMFFIToLLVMPasses();
void registerConvertTVMFFIToLLVMInterface(mlir::DialectRegistry &registry);

} // namespace libtriton::tvm_ffi

#endif // LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TVMFFITOLLVM_H_
