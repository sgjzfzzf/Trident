#ifndef LIBTRITON_CORE_CONVERSION_TORCHTOAOTINDUCTOR_TORCHTOAOTINDUCTOR_H_
#define LIBTRITON_CORE_CONVERSION_TORCHTOAOTINDUCTOR_TORCHTOAOTINDUCTOR_H_

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace libtriton::torch {

#define GEN_PASS_DECL_CONVERTTORCHTOAOTINDUCTOR
#include "libtriton-core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTTORCHTOAOTINDUCTOR
#include "libtriton-core/Conversion/Passes.h.inc"

void populateTorchToAOTInductorConversionPatterns(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns);

void registerConvertTorchToAOTInductorPass();

} // namespace libtriton::torch

#endif // LIBTRITON_CORE_CONVERSION_TORCHTOAOTINDUCTOR_TORCHTOAOTINDUCTOR_H_
