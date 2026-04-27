#ifndef LIBTRITON_CORE_CONVERSION_TRITONRTTOLLVM_TRITONRTTOLLVM_H_
#define LIBTRITON_CORE_CONVERSION_TRITONRTTOLLVM_TRITONRTTOLLVM_H_

#include "libtriton-core/Dialect/TritonRT/IR/TritonRTDialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace libtriton::triton_rt {

#define GEN_PASS_DECL_CONVERTTRITONRTTOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTTRITONRTTOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

void populateTritonRTToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns);

void registerConvertTritonRTToLLVMPass();
void registerConvertTritonRTToLLVMInterface(mlir::DialectRegistry &registry);

} // namespace libtriton::triton_rt

#endif // LIBTRITON_CORE_CONVERSION_TRITONRTTOLLVM_TRITONRTTOLLVM_H_
