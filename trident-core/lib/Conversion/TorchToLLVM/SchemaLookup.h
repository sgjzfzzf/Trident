//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_TORCHEXTTOLLVM_SCHEMALOOKUP_H_
#define TRIDENT_CORE_CONVERSION_TORCHEXTTOLLVM_SCHEMALOOKUP_H_

#include "mlir-c/IR.h"
#include "mlir-c/Rewrite.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Looks up the schema of a torch.aten.* operator via c10::Dispatcher,
/// converts each type-erased StableIValue operand to the proper type based
/// on the schema, calls the dispatcher, and converts results back to
/// type-erased StableIValue.
///
/// \param op       The MLIR operation.
/// \param operands Array of type-erased (StableIValue) operands.
/// \param results  Output array for type-erased (StableIValue) results.
/// \param rewriter The MLIR conversion pattern rewriter.
/// \return 0 on success, nonzero on failure.
int TridentSchemaDispatchTorchAtenOp(MlirOperation op, MlirValue *operands,
                                     MlirValue *results,
                                     MlirConversionPatternRewriter rewriter);

#ifdef __cplusplus
}
#endif

#endif // TRIDENT_CORE_CONVERSION_TORCHEXTTOLLVM_SCHEMALOOKUP_H_
