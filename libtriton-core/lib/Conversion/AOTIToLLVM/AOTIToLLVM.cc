#include "libtriton-core/Conversion/AOTIToLLVM/AOTIToLLVM.h"
#include "libtriton-core/Conversion/AOTIToLLVM/AOTICAPIDescriptors.h"
#include "libtriton-core/Conversion/Utils/GlobalString.h"
#include "libtriton-core/Dialect/AOTInductor/IR/AOTInductorDialect.h"
#include "libtriton-core/Dialect/AOTInductor/IR/AOTInductorOps.h"
#include "libtriton-core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"

namespace libtriton::aoti {

#define GEN_PASS_DEF_CONVERTAOTITOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Type conversion handlers
//===----------------------------------------------------------------------===//

/// CRTP base: auto-generates matches(mlir::Type) from the Torch type parameter.
template <typename TorchType> struct TypeHandlerBase {
  static bool matches(mlir::Type type) { return mlir::isa<TorchType>(type); }
};

struct BaseTensorHandler : TypeHandlerBase<mlir::torch::Torch::BaseTensorType> {
  static mlir::LogicalResult store(mlir::OpBuilder &builder, mlir::Value input,
                                   mlir::Value ptr) {
    mlir::Location loc = input.getLoc();
    mlir::Value ptrInt = mlir::LLVM::PtrToIntOp::create(
        builder, loc, mlir::IntegerType::get(builder.getContext(), 64), input);
    mlir::LLVM::StoreOp::create(builder, loc, ptrInt, ptr);
    return mlir::success();
  }
  static mlir::FailureOr<mlir::Value> load(mlir::OpBuilder &rewriter,
                                           mlir::Value loaded) {
    return mlir::LLVM::IntToPtrOp::create(
               rewriter, loaded.getLoc(),
               mlir::LLVM::LLVMPointerType::get(rewriter.getContext()), loaded)
        .getResult();
  }
};

struct BoolHandler : TypeHandlerBase<mlir::torch::Torch::BoolType> {
  static mlir::LogicalResult store(mlir::OpBuilder &builder, mlir::Value input,
                                   mlir::Value ptr) {
    mlir::Location loc = input.getLoc();
    mlir::Value ext = mlir::LLVM::ZExtOp::create(
        builder, loc, mlir::IntegerType::get(builder.getContext(), 64), input);
    mlir::LLVM::StoreOp::create(builder, loc, ext, ptr);
    return mlir::success();
  }
  static mlir::FailureOr<mlir::Value> load(mlir::OpBuilder &rewriter,
                                           mlir::Value loaded) {
    return mlir::LLVM::TruncOp::create(
               rewriter, loaded.getLoc(),
               mlir::IntegerType::get(rewriter.getContext(), 1), loaded)
        .getResult();
  }
};

struct IntHandler : TypeHandlerBase<mlir::torch::Torch::IntType> {
  static mlir::LogicalResult store(mlir::OpBuilder &builder, mlir::Value input,
                                   mlir::Value ptr) {
    mlir::LLVM::StoreOp::create(builder, input.getLoc(), input, ptr);
    return mlir::success();
  }
  static mlir::FailureOr<mlir::Value> load(mlir::OpBuilder &,
                                           mlir::Value loaded) {
    return loaded;
  }
};

struct FloatHandler : TypeHandlerBase<mlir::torch::Torch::FloatType> {
  static mlir::LogicalResult store(mlir::OpBuilder &builder, mlir::Value input,
                                   mlir::Value ptr) {
    mlir::Location loc = input.getLoc();
    mlir::Value bits = mlir::LLVM::BitcastOp::create(
        builder, loc, mlir::IntegerType::get(builder.getContext(), 64), input);
    mlir::LLVM::StoreOp::create(builder, loc, bits, ptr);
    return mlir::success();
  }
  static mlir::FailureOr<mlir::Value> load(mlir::OpBuilder &rewriter,
                                           mlir::Value loaded) {
    return mlir::LLVM::BitcastOp::create(
               rewriter, loaded.getLoc(),
               mlir::Float64Type::get(rewriter.getContext()), loaded)
        .getResult();
  }
};

struct NoneHandler : TypeHandlerBase<mlir::torch::Torch::NoneType> {
  static mlir::LogicalResult store(mlir::OpBuilder &builder, mlir::Value,
                                   mlir::Value ptr) {
    mlir::Location loc = ptr.getLoc();
    mlir::LLVM::StoreOp::create(
        builder, loc,
        mlir::LLVM::ConstantOp::create(
            builder, loc, mlir::IntegerType::get(builder.getContext(), 64), 0),
        ptr);
    return mlir::success();
  }
  static mlir::FailureOr<mlir::Value> load(mlir::OpBuilder &, mlir::Value) {
    // None → no value produced.
    return mlir::Value();
  }
};

//===----------------------------------------------------------------------===//
// Variadic dispatch: folds over handlers, short-circuits on first match
//===----------------------------------------------------------------------===//

template <typename Handler> struct HandlerCaller {
  static mlir::LogicalResult tryStore(mlir::Type type, mlir::OpBuilder &builder,
                                      mlir::Value input, mlir::Value ptr) {
    return Handler::matches(type) ? Handler::store(builder, input, ptr)
                                  : mlir::failure();
  }

  static mlir::FailureOr<mlir::Value>
  tryLoad(mlir::Type type, mlir::OpBuilder &builder, mlir::Value ptr) {
    return Handler::matches(type) ? Handler::load(builder, ptr)
                                  : mlir::FailureOr<mlir::Value>();
  }
};

template <typename... Handlers> struct TypeDispatch {
  static mlir::LogicalResult store(mlir::Type type, mlir::OpBuilder &builder,
                                   mlir::Value input, mlir::Value ptr) {
    mlir::LogicalResult result = mlir::failure();
    bool matched = (mlir::succeeded(result = HandlerCaller<Handlers>::tryStore(
                                        type, builder, input, ptr)) ||
                    ...);
    assert(matched && "no handler matched for type");
    return result;
  }

  static mlir::FailureOr<mlir::Value>
  load(mlir::Type type, mlir::OpBuilder &builder, mlir::Value ptr) {
    mlir::FailureOr<mlir::Value> result = mlir::failure();
    bool matched = (mlir::succeeded(result = HandlerCaller<Handlers>::tryLoad(
                                        type, builder, ptr)) ||
                    ...);
    assert(matched && "no handler matched for type");
    return result;
  }
};

using AllHandlers = TypeDispatch<BaseTensorHandler, BoolHandler, IntHandler,
                                 FloatHandler, NoneHandler>;

/// Converts aoti.torch_call_dispatcher to an LLVM call to
/// aoti_torch_call_dispatcher().
class ConvertTorchCallDispatcherOp
    : public mlir::OpConversionPattern<TorchCallDispatcherOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(TorchCallDispatcherOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();

    const size_t numInputs = adaptor.getInputs().size();
    const size_t numResults = op.getResults().size();
    const size_t maxCount = std::max(numInputs, numResults);

    // Allocate an i64 (uint64) slot array with maxCount elements on the stack.
    mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);
    mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
    mlir::Value slotArray = mlir::LLVM::AllocaOp::create(
        rewriter, loc, ptrTy, i64Ty,
        mlir::LLVM::ConstantOp::create(
            rewriter, loc, mlir::IntegerType::get(context, 64), maxCount)
            .getResult());

    // Store each adapted input into the corresponding slot.
    for (auto [i, pair] :
         llvm::enumerate(llvm::zip(op.getInputs(), adaptor.getInputs()))) {
      auto [origInput, adaptedInput] = pair;
      mlir::Type origType = origInput.getType();

      // Get pointer to the i-th u64 slot.
      mlir::Value slotPtr = mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy,
                                                      i64Ty, slotArray, {i});

      if (mlir::failed(
              AllHandlers::store(origType, rewriter, adaptedInput, slotPtr))) {
        return op.emitError("unsupported input type: ") << origType;
      }
    }

    // Get or create the aoti_torch_call_dispatcher function declaration.
    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return op.emitError("op is not inside a module");
    }
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> calleeOrErr =
        capi::getOrCreateaoti_torch_call_dispatcher(moduleOp);
    if (mlir::failed(calleeOrErr)) {
      return mlir::failure();
    }

    // Create global string constants for op_name and overload_name.
    mlir::Value opNamePtr = conversion::utils::getOrCreateGlobalString(
        rewriter, loc, moduleOp, "op", op.getOpName());
    mlir::Value overloadNamePtr = conversion::utils::getOrCreateGlobalString(
        rewriter, loc, moduleOp, "overload", op.getOverloadName());

    // Call aoti_torch_call_dispatcher(opName, overloadName, slotArray).
    mlir::LLVM::CallOp callOp = mlir::LLVM::CallOp::create(
        rewriter, loc, *calleeOrErr,
        mlir::ValueRange{opNamePtr, overloadNamePtr, slotArray});

    // Replace the original op with results loaded and converted from slots.
    llvm::SmallVector<mlir::Value> results;
    for (auto [i, resultType] : llvm::enumerate(op.getResultTypes())) {
      mlir::Value slotPtr = mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy,
                                                      i64Ty, slotArray, {i});
      mlir::Value loaded =
          mlir::LLVM::LoadOp::create(rewriter, loc, i64Ty, slotPtr);
      mlir::FailureOr<mlir::Value> converted =
          AllHandlers::load(resultType, rewriter, loaded);
      if (mlir::failed(converted)) {
        return op.emitError("unsupported result type: ") << resultType;
      }
      if (mlir::Value val = converted.value()) {
        results.push_back(val);
      }
    }
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

class ConvertAOTIToLLVMPass
    : public impl::ConvertAOTIToLLVMBase<ConvertAOTIToLLVMPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::ConversionTarget target(context);
    mlir::LLVMTypeConverter typeConverter(&context);
    mlir::RewritePatternSet patterns(&context);
    libtriton::torch::setupBackendTypeConversion(target, typeConverter);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct AOTIToLLVMDialectInterface : public mlir::ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(
      mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
      mlir::RewritePatternSet &patterns) const final {
    // Setup type conversion for torch types before adding patterns, so that
    // the type converter can handle Torch tensor/bool/int/float/optional etc.
    // types when patterns query adaptor types.
    libtriton::torch::setupBackendTypeConversion(target, typeConverter);
    populateAOTIToLLVMConversionPatterns(target, typeConverter, patterns);
  }
};

} // namespace

void populateAOTIToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertTorchCallDispatcherOp>(typeConverter,
                                             patterns.getContext());
  target.addIllegalDialect<AOTInductorDialect>();
  target.addLegalDialect<mlir::BuiltinDialect, mlir::LLVM::LLVMDialect>();
}

void registerConvertAOTIToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx,
                            libtriton::aoti::AOTInductorDialect *dialect) {
    dialect->addInterfaces<AOTIToLLVMDialectInterface>();
  });
}

} // namespace libtriton::aoti
