// TVMFFIToLLVM.cc - Pass that lowers TVMFFI dialect ops to LLVM dialect.
//
// Currently a scaffold: the conversion target, type converter, and rewrite
// patterns are not yet implemented. The pass is registered via the static
// `kPass` initializer (mlir::PassRegistration), so the explicit
// registerConvertTVMFFIToLLVMPass() / registerTVMFFIToLLVMPasses() functions
// are intentional no-ops kept for API consistency with the MLIR pass registry
// convention (callers can call them without knowing whether static registration
// is already in effect).

#include "libtriton_core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace libtriton::tvm_ffi {
namespace {

class ConvertTVMFFIToLLVMPass
    : public mlir::PassWrapper<ConvertTVMFFIToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTVMFFIToLLVMPass)

  llvm::StringRef getArgument() const final {
    return "convert-tvm-ffi-to-llvm";
  }

  llvm::StringRef getDescription() const final {
    return "Lower TVMFFI dialect operations to LLVM dialect";
  }

  void runOnOperation() final {
    // TODO: Add conversion target, type converter, and rewrite patterns.
  }
};

static mlir::PassRegistration<ConvertTVMFFIToLLVMPass> kPass;

} // namespace

std::unique_ptr<mlir::Pass> createConvertTVMFFIToLLVMPass() {
  return std::make_unique<ConvertTVMFFIToLLVMPass>();
}

void registerConvertTVMFFIToLLVMPass() {
  // Registration is handled by static PassRegistration above.
}

void registerTVMFFIToLLVMPasses() { registerConvertTVMFFIToLLVMPass(); }

} // namespace libtriton::tvm_ffi
