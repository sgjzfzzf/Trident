//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBTRITON_CORE_CONVERSION_TORCHEXTTOLLVM_SCHEMALOOKUP_H_
#define LIBTRITON_CORE_CONVERSION_TORCHEXTTOLLVM_SCHEMALOOKUP_H_

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
int LibTritonSchemaDispatchTorchAtenOp(MlirOperation op, MlirValue *operands,
                                       MlirValue *results,
                                       MlirConversionPatternRewriter rewriter);

#ifdef __cplusplus
}
#endif

#endif // LIBTRITON_CORE_CONVERSION_TORCHEXTTOLLVM_SCHEMALOOKUP_H_
