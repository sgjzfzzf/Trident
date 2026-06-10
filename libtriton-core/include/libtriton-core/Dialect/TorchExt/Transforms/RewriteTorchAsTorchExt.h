#ifndef LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_REWRITORCHASTORCHEXT_H_
#define LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_REWRITORCHASTORCHEXT_H_

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace libtriton::torchext {

#define GEN_PASS_DECL_REWRITETORCHASTORCHEXT
#include "libtriton-core/Dialect/TorchExt/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION_REWRITETORCHASTORCHEXT
#include "libtriton-core/Dialect/TorchExt/Transforms/Passes.h.inc"

void populateRewriteTorchAsTorchExtPatterns(mlir::RewritePatternSet &patterns);

void registerRewriteTorchAsTorchExtPass();

} // namespace libtriton::torchext

#endif // LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_REWRITORCHASTORCHEXT_H_
