#include "libtriton-core/Conversion/Bufferization/Bufferization.h"

#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace libtriton::conversion {

#define GEN_PASS_DEF_LIBTRITONONESHOTBUFFERIZE
#include "libtriton-core/Conversion/Passes.h.inc"

namespace {
static mlir::FailureOr<mlir::bufferization::LayoutMapOption>
parseLayoutMapOption(llvm::StringRef optionText) {
  if (optionText == "infer-layout-map") {
    return mlir::bufferization::LayoutMapOption::InferLayoutMap;
  } else if (optionText == "identity-layout-map") {
    return mlir::bufferization::LayoutMapOption::IdentityLayoutMap;
  } else if (optionText == "fully-dynamic-layout-map") {
    return mlir::bufferization::LayoutMapOption::FullyDynamicLayoutMap;
  } else {
    return mlir::failure();
  }
}

class LibTritonOneShotBufferizePass
    : public impl::LibTritonOneShotBufferizeBase<
          LibTritonOneShotBufferizePass> {
public:
  using Base::Base;

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                    mlir::gpu::GPUDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                    libtriton::torch_ext::TorchExtDialect>();
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    mlir::cf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  }

  void runOnOperation() final {
    mlir::ModuleOp moduleOp = getOperation();

    mlir::bufferization::OneShotBufferizationOptions options;
    options.bufferizeFunctionBoundaries = bufferizeFunctionBoundaries;

    mlir::FailureOr<mlir::bufferization::LayoutMapOption> typeConversion =
        parseLayoutMapOption(functionBoundaryTypeConversion);
    if (mlir::failed(typeConversion)) {
      getOperation().emitOpError(
          "unsupported function-boundary-type-conversion: ")
          << functionBoundaryTypeConversion;
      signalPassFailure();
      return;
    }
    options.setFunctionBoundaryTypeConversion(*typeConversion);

    mlir::bufferization::BufferizationState state;
    if (mlir::failed(mlir::bufferization::runOneShotModuleBufferize(
            moduleOp.getOperation(), options, state))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace libtriton::conversion
