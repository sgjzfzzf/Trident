#ifndef LIBTRITON_CORE_CONVERSION_TORCHTOCF_TORCHTOCF_H_
#define LIBTRITON_CORE_CONVERSION_TORCHTOCF_TORCHTOCF_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace libtriton::torch {

#define GEN_PASS_DECL_CONVERTTORCHTOCF
#include "libtriton-core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTTORCHTOCF
#include "libtriton-core/Conversion/Passes.h.inc"

void populateTorchToCfConversionPatterns(mlir::ConversionTarget &target,
                                         mlir::TypeConverter &typeConverter,
                                         mlir::RewritePatternSet &patterns);

void registerConvertTorchToCfPass();

} // namespace libtriton::torch

#endif // LIBTRITON_CORE_CONVERSION_TORCHTOCF_TORCHTOCF_H_
