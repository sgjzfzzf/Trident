#include <cstdint>

#include "libtriton-core/Analysis/MemRefOriginAnalysis/MemRefOriginAnalysis.h"
#include "libtriton-core/Dialect/DLPack/IR/DLPackDialect.h"
#include "libtriton-core/Dialect/DLPack/IR/DLPackOps.h"
#include "libtriton-core/Dialect/DLPack/IR/DLPackTypes.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/raw_ostream.h"

namespace libtriton::analysis {

static bool isMemRefType(mlir::Type type);
static MemRefOrigin
joinMemRefOperands(mlir::ArrayRef<const MemRefOriginLattice *> operands,
                   mlir::ValueRange opOperands);

// Helpers for resolveMemRefOrigin; not public API.
static std::optional<MemRefOriginKind>
readCallResultOrigin(mlir::DataFlowSolver &solver, mlir::Value value);
static bool isOriginCarrierType(mlir::Type type);
static std::optional<MemRefOriginKind>
readPassthroughOrigin(mlir::DataFlowSolver &solver, mlir::Value value);

void MemRefOrigin::print(llvm::raw_ostream &os) const {
  switch (kind) {
  case MemRefOriginKind::Unknown:
    os << "unknown";
    break;
  case MemRefOriginKind::External:
    os << "external";
    break;
  case MemRefOriginKind::InternalAlloc:
    os << "internal_alloc";
    break;
  case MemRefOriginKind::InternalAlloca:
    os << "internal_alloca";
    break;
  case MemRefOriginKind::Mixed:
    os << "mixed";
    break;
  }
}

MemRefOrigin MemRefOrigin::join(const MemRefOrigin &lhs,
                                const MemRefOrigin &rhs) {
  MemRefOrigin joined;
  if (lhs.kind == rhs.kind) {
    joined = lhs;
  } else if (lhs.kind == MemRefOriginKind::Unknown) {
    joined = rhs;
  } else if (rhs.kind == MemRefOriginKind::Unknown) {
    joined = lhs;
  } else {
    joined.kind = MemRefOriginKind::Mixed;
  }
  return joined;
}

bool MemRefOrigin::operator==(const MemRefOrigin &rhs) const {
  return kind == rhs.kind;
}

bool isMemRefType(mlir::Type type) {
  return mlir::isa<mlir::BaseMemRefType>(type);
}

bool isOriginCarrierType(mlir::Type type) {
  return isMemRefType(type) ||
         mlir::isa<libtriton::dlpack::DLTensorType>(type) ||
         mlir::isa<libtriton::dlpack::DLManagedTensorType>(type);
}

MemRefOrigin
joinMemRefOperands(mlir::ArrayRef<const MemRefOriginLattice *> operands,
                   mlir::ValueRange opOperands) {
  MemRefOrigin joined;
  for (const auto [operand, lattice] : llvm::zip(opOperands, operands)) {
    if (isMemRefType(operand.getType())) {
      joined = MemRefOrigin::join(joined, lattice->getValue());
    }
  }
  return joined;
}

std::optional<MemRefOriginKind>
readFuncArgumentOrigin(mlir::DataFlowSolver &solver, mlir::Value value) {
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    if (blockArg.getOwner() && blockArg.getOwner()->isEntryBlock() &&
        isMemRefType(value.getType())) {
      if (const auto *state = solver.lookupState<MemRefOriginLattice>(value)) {
        return state->getValue().kind;
      }
    }
  }
  return std::nullopt;
}

std::optional<MemRefOriginKind>
readCallResultOrigin(mlir::DataFlowSolver &solver, mlir::Value value) {
  if (auto callOp = value.getDefiningOp<mlir::func::CallOp>()) {
    if (auto result = mlir::dyn_cast<mlir::OpResult>(value)) {
      if (!isMemRefType(value.getType())) {
        return std::nullopt;
      }
      mlir::ModuleOp moduleOp = callOp->getParentOfType<mlir::ModuleOp>();
      if (moduleOp) {
        auto *calleeOp =
            mlir::SymbolTable::lookupSymbolIn(moduleOp, callOp.getCallee());
        if (auto callee =
                mlir::dyn_cast_if_present<mlir::func::FuncOp>(calleeOp)) {
          MemRefOrigin resultOrigin;
          callee.walk([&](mlir::func::ReturnOp returnOp) {
            if (result.getResultNumber() < returnOp.getNumOperands()) {
              mlir::Value returnedValue =
                  returnOp.getOperand(result.getResultNumber());
              MemRefOriginKind kind =
                  resolveMemRefOrigin(solver, returnedValue);
              resultOrigin =
                  MemRefOrigin::join(resultOrigin, MemRefOrigin{kind});
            }
          });
          if (resultOrigin.kind != MemRefOriginKind::Unknown) {
            return resultOrigin.kind;
          }
        }
      }
    }
  }
  return std::nullopt;
}

std::optional<MemRefOriginKind>
readPassthroughOrigin(mlir::DataFlowSolver &solver, mlir::Value value) {
  mlir::Operation *definingOp = value.getDefiningOp();
  if (!definingOp) {
    return std::nullopt;
  }

  if (definingOp->getName().getDialectNamespace() ==
          libtriton::dlpack::DLPackDialect::getDialectNamespace() &&
      definingOp->getNumResults() == 1 && definingOp->getResult(0) == value) {
    MemRefOrigin joined;
    for (mlir::Value operand : definingOp->getOperands()) {
      if (isOriginCarrierType(operand.getType())) {
        joined = MemRefOrigin::join(
            joined, MemRefOrigin{resolveMemRefOrigin(solver, operand)});
      }
    }
    return joined.kind;
  }
  return std::nullopt;
}

MemRefOriginKind resolveMemRefOrigin(mlir::DataFlowSolver &solver,
                                     mlir::Value value) {
  if (std::optional<MemRefOriginKind> kind;
      (kind = readFuncArgumentOrigin(solver, value)) ||
      (kind = readCallResultOrigin(solver, value)) ||
      (kind = readPassthroughOrigin(solver, value))) {
    return *kind;
  }
  if (const auto *state = solver.lookupState<MemRefOriginLattice>(value)) {
    return state->getValue().kind;
  }
  return MemRefOriginKind::Unknown;
}

mlir::LogicalResult MemRefOriginDataFlowAnalysis::visitOperation(
    mlir::Operation *op, mlir::ArrayRef<const MemRefOriginLattice *> operands,
    mlir::ArrayRef<MemRefOriginLattice *> results) {
  MemRefOrigin propagated = joinMemRefOperands(operands, op->getOperands());

  for (const auto [resultValue, resultLattice] :
       llvm::zip(op->getResults(), results)) {
    if (isMemRefType(resultValue.getType())) {
      MemRefOrigin resultOrigin;
      if (mlir::isa<mlir::memref::AllocOp>(op)) {
        resultOrigin.kind = MemRefOriginKind::InternalAlloc;
      } else if (mlir::isa<mlir::memref::AllocaOp>(op)) {
        resultOrigin.kind = MemRefOriginKind::InternalAlloca;
      } else {
        resultOrigin = propagated;
      }
      propagateIfChanged(resultLattice, resultLattice->join(resultOrigin));
    }
  }
  return mlir::success();
}

void MemRefOriginDataFlowAnalysis::setToEntryState(
    MemRefOriginLattice *lattice) {
  mlir::Value value = lattice->getAnchor();
  MemRefOrigin entry;
  if (mlir::BlockArgument arg = mlir::dyn_cast<mlir::BlockArgument>(value);
      isMemRefType(value.getType()) && arg && arg.getOwner()->isEntryBlock()) {
    entry.kind = MemRefOriginKind::External;
  }
  propagateIfChanged(lattice, lattice->join(entry));
}

} // namespace libtriton::analysis
