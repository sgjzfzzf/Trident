#ifndef LIBTRITON_CORE_CONVERSION_TORCHTOARITH_TORCHTOARITH_H_
#define LIBTRITON_CORE_CONVERSION_TORCHTOARITH_TORCHTOARITH_H_

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace libtriton::torch {

#define GEN_PASS_DECL_CONVERTTORCHTOARITH
#include "libtriton-core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTTORCHTOARITH
#include "libtriton-core/Conversion/Passes.h.inc"

void populateTorchToArithConversionPatterns(mlir::ConversionTarget &target,
                                            mlir::RewritePatternSet &patterns);

void registerConvertTorchToArithPass();

} // namespace libtriton::torch

#endif // LIBTRITON_CORE_CONVERSION_TORCHTOARITH_TORCHTOARITH_H_
