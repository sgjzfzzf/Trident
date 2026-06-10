#include "libtriton-core/Conversion/TorchExtToLLVM/TorchExtToLLVM.h"
#include "libtriton-core/Conversion/Utils/AOTICAPIDescriptors.h"
#include "libtriton-core/Conversion/Utils/GlobalString.h"
#include "libtriton-core/Conversion/Utils/StableCAPIDescriptors.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "libtriton-core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"

namespace libtriton::torchext {

#define GEN_PASS_DEF_CONVERTTORCHEXTTOLLVM
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
  static mlir::FailureOr<mlir::Value> to(mlir::OpBuilder &builder,
                                         mlir::Value input) {
    mlir::Location loc = input.getLoc();
    return mlir::LLVM::PtrToIntOp::create(
               builder, loc, mlir::IntegerType::get(builder.getContext(), 64),
               input)
        .getResult();
  }
  static mlir::FailureOr<mlir::Value> from(mlir::OpBuilder &rewriter,
                                           mlir::Value loaded) {
    return mlir::LLVM::IntToPtrOp::create(
               rewriter, loaded.getLoc(),
               mlir::LLVM::LLVMPointerType::get(rewriter.getContext()), loaded)
        .getResult();
  }
};

struct BoolHandler : TypeHandlerBase<mlir::torch::Torch::BoolType> {
  static mlir::FailureOr<mlir::Value> to(mlir::OpBuilder &builder,
                                         mlir::Value input) {
    mlir::Location loc = input.getLoc();
    return mlir::LLVM::ZExtOp::create(
               builder, loc, mlir::IntegerType::get(builder.getContext(), 64),
               input)
        .getResult();
  }
  static mlir::FailureOr<mlir::Value> from(mlir::OpBuilder &rewriter,
                                           mlir::Value loaded) {
    return mlir::LLVM::TruncOp::create(
               rewriter, loaded.getLoc(),
               mlir::IntegerType::get(rewriter.getContext(), 1), loaded)
        .getResult();
  }
};

struct IntHandler : TypeHandlerBase<mlir::torch::Torch::IntType> {
  static mlir::FailureOr<mlir::Value> to(mlir::OpBuilder &, mlir::Value input) {
    return input;
  }
  static mlir::FailureOr<mlir::Value> from(mlir::OpBuilder &,
                                           mlir::Value loaded) {
    return loaded;
  }
};

struct FloatHandler : TypeHandlerBase<mlir::torch::Torch::FloatType> {
  static mlir::FailureOr<mlir::Value> to(mlir::OpBuilder &builder,
                                         mlir::Value input) {
    mlir::Location loc = input.getLoc();
    return mlir::LLVM::BitcastOp::create(
               builder, loc, mlir::IntegerType::get(builder.getContext(), 64),
               input)
        .getResult();
  }
  static mlir::FailureOr<mlir::Value> from(mlir::OpBuilder &rewriter,
                                           mlir::Value loaded) {
    return mlir::LLVM::BitcastOp::create(
               rewriter, loaded.getLoc(),
               mlir::Float64Type::get(rewriter.getContext()), loaded)
        .getResult();
  }
};

struct NoneHandler : TypeHandlerBase<mlir::torch::Torch::NoneType> {
  static mlir::FailureOr<mlir::Value> to(mlir::OpBuilder &builder,
                                         mlir::Value input) {
    return mlir::LLVM::ConstantOp::create(
               builder, input.getLoc(),
               mlir::IntegerType::get(builder.getContext(), 64), 0)
        .getResult();
  }
  static mlir::FailureOr<mlir::Value> from(mlir::OpBuilder &, mlir::Value) {
    // None → no value produced.
    return mlir::Value();
  }
};

//===----------------------------------------------------------------------===//
// Variadic dispatch: folds over handlers, short-circuits on first match
//===----------------------------------------------------------------------===//

template <typename Handler> struct HandlerCaller {
  static mlir::FailureOr<mlir::Value>
  tryTo(mlir::Type type, mlir::OpBuilder &builder, mlir::Value input) {
    return Handler::matches(type) ? Handler::to(builder, input)
                                  : mlir::FailureOr<mlir::Value>();
  }

  static mlir::FailureOr<mlir::Value>
  tryFrom(mlir::Type type, mlir::OpBuilder &builder, mlir::Value ptr) {
    return Handler::matches(type) ? Handler::from(builder, ptr)
                                  : mlir::FailureOr<mlir::Value>();
  }
};

template <typename... Handlers> struct TypeDispatch {
  static mlir::FailureOr<mlir::Value>
  to(mlir::Type type, mlir::OpBuilder &builder, mlir::Value input) {
    mlir::FailureOr<mlir::Value> result = mlir::failure();
    (mlir::succeeded(
         result = HandlerCaller<Handlers>::tryTo(type, builder, input)) ||
     ...);
    return result;
  }

  static mlir::FailureOr<mlir::Value>
  from(mlir::Type type, mlir::OpBuilder &builder, mlir::Value ptr) {
    mlir::FailureOr<mlir::Value> result = mlir::failure();
    (mlir::succeeded(
         result = HandlerCaller<Handlers>::tryFrom(type, builder, ptr)) ||
     ...);
    return result;
  }
};

using AllHandlers = TypeDispatch<BaseTensorHandler, BoolHandler, IntHandler,
                                 FloatHandler, NoneHandler>;

/// Converts torchext.call_dispatcher to an LLVM call to
/// aoti_torch_call_dispatcher().
class ConvertCallDispatcherOp
    : public mlir::OpConversionPattern<CallDispatcherOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CallDispatcherOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();

    const size_t numInputs = adaptor.getInputs().size();
    const size_t numResults = op.getResults().size();
    const size_t maxCount = std::max(numInputs, numResults);

    // Allocate an i64 (uint64) slot array with maxCount elements on the stack.
    mlir::LLVM::LLVMPointerType ptrTy =
        mlir::LLVM::LLVMPointerType::get(context);
    mlir::IntegerType i64Ty = mlir::IntegerType::get(context, 64);
    mlir::Value array = mlir::LLVM::AllocaOp::create(
        rewriter, loc, ptrTy, i64Ty,
        mlir::LLVM::ConstantOp::create(
            rewriter, loc, mlir::IntegerType::get(context, 64), maxCount)
            .getResult());

    // Store each adapted input into the corresponding slot.
    for (auto [i, pair] :
         llvm::enumerate(llvm::zip(op.getInputs(), adaptor.getInputs()))) {
      auto [origInput, adaptedInput] = pair;
      mlir::Type origType = origInput.getType();

      // Convert adapted input to i64 (type-erased StableIValue).
      mlir::FailureOr<mlir::Value> ival =
          AllHandlers::to(origType, rewriter, adaptedInput);
      if (mlir::failed(ival)) {
        return op.emitError("unsupported input type: ") << origType;
      }

      // Store the i64 into the corresponding slot.
      mlir::Value ptr =
          mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, i64Ty, array, {i});
      mlir::LLVM::StoreOp::create(rewriter, loc, ival.value(), ptr);
    }

    // Get or create the aoti_torch_call_dispatcher function declaration.
    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return op.emitError("op is not inside a module");
    }
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> calleeOrErr =
        libtriton::conversion::utils::getOrCreateaoti_torch_call_dispatcher(
            moduleOp);
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
        mlir::ValueRange{opNamePtr, overloadNamePtr, array});

    // Replace the original op with results loaded and converted from slots.
    llvm::SmallVector<mlir::Value> results;
    for (auto [i, type] : llvm::enumerate(op.getResultTypes())) {
      mlir::Value ptr =
          mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, i64Ty, array, {i});
      mlir::Value loaded =
          mlir::LLVM::LoadOp::create(rewriter, loc, i64Ty, ptr);
      mlir::FailureOr<mlir::Value> converted =
          AllHandlers::from(type, rewriter, loaded);
      if (mlir::failed(converted)) {
        return op.emitError("unsupported result type: ") << type;
      }
      if (mlir::Value val = converted.value()) {
        results.push_back(val);
      }
    }
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// StableList operations
//===----------------------------------------------------------------------===//

/// Converts torch.prim.ListConstruct to torch_new_list_reserve_size() +
/// torch_list_push_back() LLVM calls.
class ConvertPrimListConstructOp
    : public mlir::OpConversionPattern<
          mlir::torch::Torch::PrimListConstructOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::PrimListConstructOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = op.getContext();
    mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
    mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);

    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return op.emitError("op is not inside a module");
    }

    int64_t numElements = static_cast<int64_t>(op.getElements().size());

    // Allocate a stack slot for the StableListHandle* output.
    mlir::Value listPtr = mlir::LLVM::AllocaOp::create(
        rewriter, loc, ptrTy, ptrTy,
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1));

    // Call torch_new_list_reserve_size(numElements, &listHandle).
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> newListCallee =
        libtriton::conversion::utils::getOrCreateTorchNewListReserveSize(
            moduleOp);
    if (mlir::failed(newListCallee)) {
      return mlir::failure();
    }
    mlir::Value numElementsVal =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, numElements);
    mlir::LLVM::CallOp::create(rewriter, loc, *newListCallee,
                               mlir::ValueRange{numElementsVal, listPtr});

    // Load the newly created list handle.
    mlir::Value listHandle =
        mlir::LLVM::LoadOp::create(rewriter, loc, ptrTy, listPtr);

    // Get or create torch_list_push_back.
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> pushBackCallee =
        libtriton::conversion::utils::getOrCreateTorchListPushBack(moduleOp);
    if (mlir::failed(pushBackCallee)) {
      return mlir::failure();
    }

    // Push each element into the list as a type-erased StableIValue.
    for (auto pair : llvm::zip(op.getElements(), adaptor.getElements())) {
      mlir::Value origElem = std::get<0>(pair);
      mlir::Value adaptedElem = std::get<1>(pair);

      mlir::FailureOr<mlir::Value> ival =
          AllHandlers::to(origElem.getType(), rewriter, adaptedElem);
      if (mlir::failed(ival)) {
        return op.emitError("unsupported element type");
      }

      mlir::LLVM::CallOp::create(rewriter, loc, *pushBackCallee,
                                 mlir::ValueRange{listHandle, ival.value()});
    }

    rewriter.replaceOp(op, listHandle);
    return mlir::success();
  }
};

/// Converts torchext.aoti.ListDeleteList to torch_delete_list() LLVM call.
class ConvertListDeleteListOp
    : public mlir::OpConversionPattern<ListDeleteListOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ListDeleteListOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = op.getContext();

    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return op.emitError("op is not inside a module");
    }

    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> calleeOrErr =
        libtriton::conversion::utils::getOrCreateTorchDeleteList(moduleOp);
    if (mlir::failed(calleeOrErr)) {
      return mlir::failure();
    }

    mlir::LLVM::CallOp::create(rewriter, loc, *calleeOrErr,
                               mlir::ValueRange{adaptor.getList()});
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ConvertTorchExtToLLVMPass
    : public impl::ConvertTorchExtToLLVMBase<ConvertTorchExtToLLVMPass> {
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

struct TorchExtToLLVMDialectInterface
    : public mlir::ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(
      mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
      mlir::RewritePatternSet &patterns) const final {
    // Setup type conversion for torch types before adding patterns, so that
    // the type converter can handle Torch tensor/bool/int/float/optional etc.
    // types when patterns query adaptor types.
    libtriton::torch::setupBackendTypeConversion(target, typeConverter);
    populateTorchExtToLLVMConversionPatterns(target, typeConverter, patterns);
  }
};

} // namespace

void populateTorchExtToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertCallDispatcherOp>(typeConverter, patterns.getContext());
  patterns.add<ConvertPrimListConstructOp, ConvertListDeleteListOp>(
      typeConverter, patterns.getContext());
  target.addIllegalOp<mlir::torch::Torch::PrimListConstructOp>();
  target.addIllegalDialect<TorchExtDialect>();
  target.addLegalDialect<mlir::BuiltinDialect, mlir::LLVM::LLVMDialect>();
}

void registerConvertTorchExtToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx,
                            libtriton::torchext::TorchExtDialect *dialect) {
    dialect->addInterfaces<TorchExtToLLVMDialectInterface>();
  });
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::torch::Torch::TorchDialect *dialect) {
        dialect->addInterfaces<TorchExtToLLVMDialectInterface>();
      });
}

} // namespace libtriton::torchext
