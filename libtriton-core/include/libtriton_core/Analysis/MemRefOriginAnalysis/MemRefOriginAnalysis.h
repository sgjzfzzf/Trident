#ifndef LIBTRITON_CORE_ANALYSIS_MEMREFORIGINANALYSIS_MEMREFORIGINANALYSIS_H_
#define LIBTRITON_CORE_ANALYSIS_MEMREFORIGINANALYSIS_MEMREFORIGINANALYSIS_H_

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace llvm {
class raw_ostream;
}

namespace libtriton::analysis {

enum class MemRefOriginKind {
  Unknown,
  External,
  InternalAlloc,
  InternalAlloca,
  Mixed,
};

struct MemRefOrigin {
  MemRefOriginKind kind = MemRefOriginKind::Unknown;

  void print(llvm::raw_ostream &os) const;
  static MemRefOrigin join(const MemRefOrigin &lhs, const MemRefOrigin &rhs);
  bool operator==(const MemRefOrigin &rhs) const;
};

class MemRefOriginLattice : public mlir::dataflow::Lattice<MemRefOrigin> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemRefOriginLattice)
  using Lattice::Lattice;
};

class MemRefOriginDataFlowAnalysis
    : public mlir::dataflow::SparseForwardDataFlowAnalysis<
          MemRefOriginLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  mlir::LogicalResult
  visitOperation(mlir::Operation *op,
                 mlir::ArrayRef<const MemRefOriginLattice *> operands,
                 mlir::ArrayRef<MemRefOriginLattice *> results) override;

  void setToEntryState(MemRefOriginLattice *lattice) override;
};

MemRefOriginKind resolveMemRefOrigin(mlir::DataFlowSolver &solver,
                                     mlir::Value value);

} // namespace libtriton::analysis

#endif // LIBTRITON_CORE_ANALYSIS_MEMREFORIGINANALYSIS_MEMREFORIGINANALYSIS_H_
