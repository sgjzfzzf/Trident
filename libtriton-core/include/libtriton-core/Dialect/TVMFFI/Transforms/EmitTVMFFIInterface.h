#ifndef LIBTRITON_CORE_DIALECT_TVMFFI_TRANSFORMS_EMITTVMFFIINTERFACE_H_
#define LIBTRITON_CORE_DIALECT_TVMFFI_TRANSFORMS_EMITTVMFFIINTERFACE_H_

#include "libtriton-core/Dialect/TVMFFI/Transforms/Passes.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace libtriton::tvm_ffi {

/// Build a TVM FFI packed-call wrapper for @p targetFunc inside @p moduleOp.
/// The wrapper is named `__tvm_ffi_<sym_name>` and follows the TVM stable C
/// ABI: `(void*, void*, int32, void*) -> int32`.
///
/// @p solver must already have been initialized and run so that
/// MemRefOriginDataFlowAnalysis results are available.
mlir::FailureOr<mlir::func::FuncOp>
buildEmitTVMFFIInterfaceWrapper(mlir::ModuleOp moduleOp,
                                mlir::DataFlowSolver &solver,
                                mlir::func::FuncOp targetFunc);

} // namespace libtriton::tvm_ffi

#endif // LIBTRITON_CORE_DIALECT_TVMFFI_TRANSFORMS_EMITTVMFFIINTERFACE_H_
