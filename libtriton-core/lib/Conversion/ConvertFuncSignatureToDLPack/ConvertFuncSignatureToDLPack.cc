// ConvertFuncSignatureToDLPack.cc - Rewrites memref function signatures.

#include <cstdint>

#include "libtriton_core/Conversion/ConvertFuncSignatureToDLPack/ConvertFuncSignatureToDLPack.h"
#include "libtriton_core/Dialect/DLPack/IR/DLPackDialect.h"
#include "libtriton_core/Dialect/DLPack/IR/DLPackOps.h"
#include "libtriton_core/Dialect/DLPack/IR/DLPackTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace libtriton::dlpack {
namespace {

bool isFromFunctionSignatureMemRef(mlir::Value value) {
  if (mlir::isa<mlir::BaseMemRefType>(value.getType())) {
    if (mlir::BlockArgument blockArg =
            mlir::dyn_cast<mlir::BlockArgument>(value)) {
      mlir::Operation *parentOp = blockArg.getOwner()->getParentOp();
      mlir::func::FuncOp parentFunc =
          mlir::dyn_cast_or_null<mlir::func::FuncOp>(parentOp);
      if (!parentFunc) {
        return false;
      }
      std::uint32_t argIndex = blockArg.getArgNumber();
      return argIndex < parentFunc.getFunctionType().getNumInputs() &&
             mlir::isa<mlir::BaseMemRefType>(
                 parentFunc.getFunctionType().getInput(argIndex));
    }

    if (libtriton::dlpack::ToMemRefOp toMemRefOp =
            value.getDefiningOp<libtriton::dlpack::ToMemRefOp>()) {
      mlir::Value tensorInput = toMemRefOp.getInput();
      if (mlir::BlockArgument blockArg =
              mlir::dyn_cast<mlir::BlockArgument>(tensorInput)) {
        return blockArg.getOwner()->isEntryBlock();
      }
      return isFromFunctionSignatureMemRef(tensorInput);
    }

    if (mlir::func::CallOp callOp = value.getDefiningOp<mlir::func::CallOp>()) {
      for (mlir::Value operand : callOp.getOperands()) {
        if (isFromFunctionSignatureMemRef(operand)) {
          return true;
        }
      }
      return false;
    }
  }

  if (libtriton::dlpack::FromMemRefBorrowedOp borrowedOp =
          value.getDefiningOp<libtriton::dlpack::FromMemRefBorrowedOp>()) {
    return isFromFunctionSignatureMemRef(borrowedOp.getInput());
  }

  if (libtriton::dlpack::ViewOp viewOp =
          value.getDefiningOp<libtriton::dlpack::ViewOp>()) {
    return isFromFunctionSignatureMemRef(viewOp.getInput());
  }

  return false;
}

mlir::Value materializeDLPackBridge(mlir::OpBuilder &builder,
                                    mlir::Type resultType,
                                    mlir::ValueRange inputs,
                                    mlir::Location loc) {
  if (inputs.size() != 1) {
    return {};
  }

  mlir::Value input = inputs[0];
  mlir::Type inputType = input.getType();

  if (mlir::isa<mlir::BaseMemRefType>(resultType) &&
      mlir::isa<libtriton::dlpack::DLTensorType>(inputType)) {
    return libtriton::dlpack::ToMemRefOp::create(builder, loc, resultType,
                                                 input)
        .getOutput();
  }

  if (mlir::isa<libtriton::dlpack::DLTensorType>(resultType) &&
      mlir::isa<mlir::BaseMemRefType>(inputType)) {
    if (isFromFunctionSignatureMemRef(input)) {
      return libtriton::dlpack::FromMemRefBorrowedOp::create(builder, loc,
                                                             resultType, input)
          .getOutput();
    }
    mlir::Type managedTensorType =
        libtriton::dlpack::DLManagedTensorType::get(builder.getContext());
    libtriton::dlpack::FromMemRefOwnedOp ownedOp =
        libtriton::dlpack::FromMemRefOwnedOp::create(builder, loc,
                                                     managedTensorType, input);
    return libtriton::dlpack::ViewOp::create(builder, loc, resultType,
                                             ownedOp.getOutput())
        .getOutput();
  }

  if (mlir::isa<libtriton::dlpack::DLTensorType>(resultType) &&
      mlir::isa<libtriton::dlpack::DLManagedTensorType>(inputType)) {
    return libtriton::dlpack::ViewOp::create(builder, loc, resultType, input)
        .getOutput();
  }

  if (mlir::isa<mlir::BaseMemRefType>(resultType) &&
      mlir::isa<libtriton::dlpack::DLManagedTensorType>(inputType)) {
    mlir::Type tensorType =
        libtriton::dlpack::DLTensorType::get(builder.getContext());
    libtriton::dlpack::ViewOp viewOp =
        libtriton::dlpack::ViewOp::create(builder, loc, tensorType, input);
    return libtriton::dlpack::ToMemRefOp::create(builder, loc, resultType,
                                                 viewOp.getOutput())
        .getOutput();
  }

  if (mlir::isa<libtriton::dlpack::DLManagedTensorType>(resultType) &&
      mlir::isa<mlir::BaseMemRefType>(inputType)) {
    return libtriton::dlpack::FromMemRefOwnedOp::create(builder, loc,
                                                        resultType, input)
        .getOutput();
  }

  return mlir::UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                  input)
      .getResult(0);
}

class ConvertFuncSignatureToDLPackPass
    : public mlir::PassWrapper<ConvertFuncSignatureToDLPackPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertFuncSignatureToDLPackPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry
        .insert<mlir::func::FuncDialect, libtriton::dlpack::DLPackDialect>();
  }

  llvm::StringRef getArgument() const final {
    return "convert-func-signature-to-dlpack";
  }

  llvm::StringRef getDescription() const final {
    return "Rewrite memref signatures to DLPack tensor signatures with "
           "OpConversion";
  }

  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::TypeConverter typeConverter;

    typeConverter.addConversion([&](mlir::Type type) -> mlir::Type {
      if (mlir::isa<mlir::BaseMemRefType>(type)) {
        return libtriton::dlpack::DLTensorType::get(type.getContext());
      }
      return type;
    });

    typeConverter.addSourceMaterialization(materializeDLPackBridge);
    typeConverter.addTargetMaterialization(materializeDLPackBridge);

    mlir::ConversionTarget target(context);
    target.addLegalDialect<mlir::BuiltinDialect,
                           libtriton::dlpack::DLPackDialect>();
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                 typeConverter.isLegal(&op.getBody());
        });
    target.addDynamicallyLegalOp<mlir::func::CallOp>(
        [&](mlir::func::CallOp op) {
          return typeConverter.isLegal(op.getResultTypes()) &&
                 typeConverter.isLegal(op.getOperandTypes());
        });
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::Operation *op) {
          return mlir::isLegalForReturnOpTypeConversionPattern(op,
                                                               typeConverter);
        });
    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation *) { return true; });

    mlir::RewritePatternSet patterns(&context);
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, typeConverter);
    mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
    mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

static mlir::PassRegistration<ConvertFuncSignatureToDLPackPass>
    kConvertFuncSignatureToDLPackPass;

} // namespace

std::unique_ptr<mlir::Pass> createConvertFuncSignatureToDLPackPass() {
  return std::make_unique<ConvertFuncSignatureToDLPackPass>();
}

void registerConvertFuncSignatureToDLPackPass() {
  // Registration is handled by static PassRegistration above.
}

} // namespace libtriton::dlpack
