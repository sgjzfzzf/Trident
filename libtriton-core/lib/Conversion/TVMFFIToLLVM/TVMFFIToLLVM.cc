//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "libtriton-core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "libtriton-core/Conversion/Utils/AOTICAPIDescriptors.h"
#include "libtriton-core/Conversion/Utils/GlobalString.h"
#include "libtriton-core/Conversion/Utils/LibTritonCAPIDescriptors.h"
#include "libtriton-core/Conversion/Utils/StdLibCAPIDescriptors.h"
#include "libtriton-core/Conversion/Utils/TVMFFICAPIDescriptors.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "libtriton-core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"

namespace libtriton::tvm_ffi {

#define GEN_PASS_DEF_CONVERTTVMFFITOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

namespace {

class Aux {};

static mlir::LLVM::LLVMStructType getTVMFFIAnyType(mlir::MLIRContext *context) {
  mlir::IntegerType i32Ty = mlir::IntegerType::get(context, 32);
  mlir::IntegerType i64Ty = mlir::IntegerType::get(context, 64);
  return mlir::LLVM::LLVMStructType::getLiteral(context, {i32Ty, i32Ty, i64Ty},
                                                true);
}

//===----------------------------------------------------------------------===//
// Type conversion handlers for packing/unpacking TVM FFI arguments
//===----------------------------------------------------------------------===//

/// CRTP base: auto-generates matches(mlir::Type) from the Torch type parameter.
template <typename TorchType> struct TypeHandlerBase {
  static bool matches(mlir::Type type) { return mlir::isa<TorchType>(type); }
};

struct BaseTensorHandler : TypeHandlerBase<mlir::torch::Torch::BaseTensorType> {
  /// The input is already a TVMFFIObjectHandle (!llvm.ptr) from the function
  /// body (via TorchExtToLLVM lowering which keeps values as handles).
  /// Just return it directly — ConvertFuncOp wraps it into a TVMFFIAny struct.
  static mlir::FailureOr<mlir::Value> to(mlir::OpBuilder &builder,
                                         mlir::Value input) {
    return input;
  }

  /// Extracts the TVMFFIObjectHandle (v_obj) from the TVMFFIAny struct and
  /// returns it as a !llvm.ptr. No conversion to AtenTensorHandle here;
  /// that happens when the handle reaches an ATen op.
  static mlir::FailureOr<mlir::Value> from(mlir::OpBuilder &builder,
                                           mlir::Value ptr, Aux &aux) {
    mlir::Location loc = ptr.getLoc();
    mlir::MLIRContext *context = builder.getContext();
    mlir::LLVM::LLVMPointerType ptrTy =
        mlir::LLVM::LLVMPointerType::get(context);

    // Extract the TVMFFIObjectHandle (v_obj) from the TVMFFIAny struct.
    // v_obj is at field index 2 in the {i32, i32, i64} struct.
    mlir::Value vobj = mlir::LLVM::GEPOp::create(
        builder, loc, ptrTy, getTVMFFIAnyType(context), ptr,
        mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    mlir::Value i64 = mlir::LLVM::LoadOp::create(
        builder, loc, mlir::IntegerType::get(context, 64), vobj);
    return mlir::LLVM::IntToPtrOp::create(builder, loc, ptrTy, i64).getResult();
  }
};

struct BoolHandler : TypeHandlerBase<mlir::torch::Torch::BoolType> {
  static mlir::FailureOr<mlir::Value> to(mlir::OpBuilder &builder,
                                         mlir::Value input) {
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(ctx);
    mlir::Value result =
        mlir::LLVM::UndefOp::create(builder, input.getLoc(), anyTy);
    mlir::Value typeIdx = mlir::LLVM::ConstantOp::create(
        builder, input.getLoc(), mlir::IntegerType::get(ctx, 32), kTVMFFIBool);
    result = mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                               result, typeIdx,
                                               llvm::ArrayRef<int64_t>{0})
                 .getResult();
    mlir::Value pl =
        mlir::LLVM::ZExtOp::create(builder, input.getLoc(),
                                   mlir::IntegerType::get(ctx, 64), input)
            .getResult();
    return mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                             result, pl,
                                             llvm::ArrayRef<int64_t>{2})
        .getResult();
  }
  static mlir::FailureOr<mlir::Value> from(mlir::OpBuilder &builder,
                                           mlir::Value ptr, Aux &) {
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Value payloadPtr = mlir::LLVM::GEPOp::create(
        builder, ptr.getLoc(), mlir::LLVM::LLVMPointerType::get(ctx),
        getTVMFFIAnyType(ctx), ptr, mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    mlir::Value loaded = mlir::LLVM::LoadOp::create(
        builder, ptr.getLoc(), mlir::IntegerType::get(ctx, 64), payloadPtr);
    return mlir::LLVM::TruncOp::create(
               builder, loaded.getLoc(),
               mlir::IntegerType::get(builder.getContext(), 1), loaded)
        .getResult();
  }
};

struct IntHandler : TypeHandlerBase<mlir::torch::Torch::IntType> {
  static mlir::FailureOr<mlir::Value> to(mlir::OpBuilder &builder,
                                         mlir::Value input) {
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(ctx);
    mlir::Value result =
        mlir::LLVM::UndefOp::create(builder, input.getLoc(), anyTy);
    mlir::Value typeIdx = mlir::LLVM::ConstantOp::create(
        builder, input.getLoc(), mlir::IntegerType::get(ctx, 32), kTVMFFIInt);
    result = mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                               result, typeIdx,
                                               llvm::ArrayRef<int64_t>{0})
                 .getResult();
    return mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                             result, input,
                                             llvm::ArrayRef<int64_t>{2})
        .getResult();
  }
  static mlir::FailureOr<mlir::Value> from(mlir::OpBuilder &builder,
                                           mlir::Value ptr, Aux &) {
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Value payloadPtr = mlir::LLVM::GEPOp::create(
        builder, ptr.getLoc(), mlir::LLVM::LLVMPointerType::get(ctx),
        getTVMFFIAnyType(ctx), ptr, mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    return mlir::LLVM::LoadOp::create(builder, ptr.getLoc(),
                                      mlir::IntegerType::get(ctx, 64),
                                      payloadPtr)
        .getResult();
  }
};

struct FloatHandler : TypeHandlerBase<mlir::torch::Torch::FloatType> {
  static mlir::FailureOr<mlir::Value> to(mlir::OpBuilder &builder,
                                         mlir::Value input) {
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(ctx);
    mlir::Value result =
        mlir::LLVM::UndefOp::create(builder, input.getLoc(), anyTy);
    mlir::Value typeIdx = mlir::LLVM::ConstantOp::create(
        builder, input.getLoc(), mlir::IntegerType::get(ctx, 32), kTVMFFIFloat);
    result = mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                               result, typeIdx,
                                               llvm::ArrayRef<int64_t>{0})
                 .getResult();
    mlir::Value pl =
        mlir::LLVM::BitcastOp::create(builder, input.getLoc(),
                                      mlir::IntegerType::get(ctx, 64), input)
            .getResult();
    return mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                             result, pl,
                                             llvm::ArrayRef<int64_t>{2})
        .getResult();
  }
  static mlir::FailureOr<mlir::Value> from(mlir::OpBuilder &builder,
                                           mlir::Value ptr, Aux &) {
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Value payloadPtr = mlir::LLVM::GEPOp::create(
        builder, ptr.getLoc(), mlir::LLVM::LLVMPointerType::get(ctx),
        getTVMFFIAnyType(ctx), ptr, mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    mlir::Value loaded = mlir::LLVM::LoadOp::create(
        builder, ptr.getLoc(), mlir::IntegerType::get(ctx, 64), payloadPtr);
    return mlir::LLVM::BitcastOp::create(
               builder, loaded.getLoc(),
               mlir::Float64Type::get(builder.getContext()), loaded)
        .getResult();
  }
};

struct NoneHandler : TypeHandlerBase<mlir::torch::Torch::NoneType> {
  static mlir::FailureOr<mlir::Value> to(mlir::OpBuilder &builder,
                                         mlir::Value input) {
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(ctx);
    mlir::Value result =
        mlir::LLVM::UndefOp::create(builder, input.getLoc(), anyTy);
    mlir::Value typeIdx = mlir::LLVM::ConstantOp::create(
        builder, input.getLoc(), mlir::IntegerType::get(ctx, 32), kTVMFFINone);
    result = mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                               result, typeIdx,
                                               llvm::ArrayRef<int64_t>{0})
                 .getResult();
    mlir::Value zero = mlir::LLVM::ConstantOp::create(
        builder, input.getLoc(), mlir::IntegerType::get(ctx, 64), 0);
    return mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                             result, zero,
                                             llvm::ArrayRef<int64_t>{2})
        .getResult();
  }
  static mlir::FailureOr<mlir::Value> from(mlir::OpBuilder &, mlir::Value,
                                           Aux &) {
    return mlir::Value();
  }
};

//===----------------------------------------------------------------------===//
// Variadic dispatch: folds over handlers, short-circuits on first match
//===----------------------------------------------------------------------===//

// ── Value-based dispatch (all handlers' to() return a value, no ptr) ──
template <typename Handler> struct HandlerCaller {
  static mlir::FailureOr<mlir::Value>
  tryTo(mlir::Type type, mlir::OpBuilder &builder, mlir::Value input) {
    return Handler::matches(type) ? Handler::to(builder, input)
                                  : mlir::FailureOr<mlir::Value>();
  }

  static mlir::FailureOr<mlir::Value> tryFrom(mlir::Type type,
                                              mlir::OpBuilder &builder,
                                              mlir::Value ptr, Aux &aux) {
    return Handler::matches(type) ? Handler::from(builder, ptr, aux)
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
  from(mlir::Type type, mlir::OpBuilder &builder, mlir::Value ptr, Aux &aux) {
    mlir::FailureOr<mlir::Value> result = mlir::failure();
    (mlir::succeeded(
         result = HandlerCaller<Handlers>::tryFrom(type, builder, ptr, aux)) ||
     ...);
    return result;
  }
};

using AllHandlers = TypeDispatch<BaseTensorHandler, BoolHandler, IntHandler,
                                 FloatHandler, NoneHandler>;

/// Converts tvm_ffi.func to func.func by converting the function type
/// signature through the type converter and inlining the region body.
class ConvertFuncOp : public mlir::OpConversionPattern<FuncOp> {
public:
  ConvertFuncOp(mlir::LLVMTypeConverter &typeConverter,
                mlir::MLIRContext *context)
      : mlir::OpConversionPattern<FuncOp>(typeConverter, context) {}

  mlir::LogicalResult
  matchAndRewrite(FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = rewriter.getContext();

    // Without support for tvm_ffi.Array, at most one return value is allowed.
    assert(op.getFunctionType().getResults().size() <= 1 &&
           "Without the support of `tvm.ffi.Array`, we only support at most "
           "one return value temporarily.");

    // TVM-FFI C ABI: int32_t(void*, void*, int32_t, void*)
    mlir::IntegerType i32Ty = rewriter.getIntegerType(32);
    mlir::LLVM::LLVMPointerType ptrTy =
        mlir::LLVM::LLVMPointerType::get(context);
    mlir::LLVM::LLVMFunctionType funcType =
        mlir::LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, ptrTy, i32Ty, ptrTy});

    // Create the LLVM function, then build all blocks from scratch.
    std::string tvmffiFuncName =
        llvm::formatv("__tvm_ffi_{0}", op.getSymName());
    mlir::LLVM::LLVMFuncOp llvmFunc = mlir::LLVM::LLVMFuncOp::create(
        rewriter, loc, tvmffiFuncName, funcType, mlir::LLVM::Linkage::External);
    mlir::Region &region = llvmFunc.getBody();
    mlir::IRMapping mapping;

    // Step 1: Create the entry block with ABI arguments from the function type.
    // TVM-FFI C ABI: int32_t(void* handle, TVMFFIAny* args, int32_t num_args,
    // TVMFFIAny* result)
    mlir::Block *entryBlock = llvmFunc.addEntryBlock(rewriter);
    mlir::Value argsPtr = entryBlock->getArgument(1); // TVMFFIAny* args
    mlir::Value retPtr = entryBlock->getArgument(3);  // TVMFFIAny* result

    // Step 2: Parse arguments. Type checks may split the entry block,
    // producing error/continue blocks that naturally precede the body.
    rewriter.setInsertionPointToStart(entryBlock);
    mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(context);
    Aux aux;
    mlir::Block &entry = op.getBody().front();
    for (auto [i, arg] : llvm::enumerate(entry.getArguments())) {
      mlir::Value slot =
          mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, anyTy, argsPtr,
                                    mlir::ArrayRef<mlir::LLVM::GEPArg>{i});
      mlir::Type argTy = arg.getType();
      mlir::FailureOr<mlir::Value> loaded =
          AllHandlers::from(argTy, rewriter, slot, aux);
      if (mlir::failed(loaded)) {
        return op.emitError("unsupported input type: ") << argTy;
      }
      mlir::Value casted = mlir::UnrealizedConversionCastOp::create(
                               rewriter, loc, argTy, *loaded)
                               .getResult(0);
      mapping.map(arg, casted);
    }

    // Step 3: Create mapped body blocks (placed after any split blocks).
    mlir::Block *firstBodyBlock = nullptr;
    for (mlir::Block &blk : op.getBody()) {
      mlir::ConversionPatternRewriter::InsertionGuard guard(rewriter);
      mlir::Block *newBlock = rewriter.createBlock(&region);
      if (firstBodyBlock == nullptr) {
        firstBodyBlock = newBlock;
      }
      mapping.map(&blk, newBlock);
    }

    // Step 4: Branch from the merge block to the first body block.
    assert(firstBodyBlock && "expected at least one block in function body");
    mlir::LLVM::BrOp::create(rewriter, loc, firstBodyBlock);

    // Step 5: Clone original function body ops into the mapped body blocks.
    for (mlir::Block &blk : op.getBody()) {
      mlir::Block *dest = mapping.lookupOrDefault(&blk);
      rewriter.setInsertionPointToEnd(dest);
      for (mlir::Operation &operation : llvm::make_early_inc_range(blk)) {
        if (ReturnOp returnOp = mlir::dyn_cast<ReturnOp>(&operation)) {
          for (mlir::Value operand : returnOp.getOperands()) {
            mlir::Value retVal = mapping.lookupOrDefault(operand);
            mlir::Type operandTy = operand.getType();
            mlir::Value casted =
                mlir::UnrealizedConversionCastOp::create(
                    rewriter, loc, getTypeConverter()->convertType(operandTy),
                    retVal)
                    .getResult(0);
            mlir::FailureOr<mlir::Value> result =
                AllHandlers::to(operandTy, rewriter, casted);
            if (mlir::failed(result)) {
              return op.emitError("unsupported return type: ") << operandTy;
            }
            // For tensor types, the handler returns a TVMFFIObjectHandle
            // (!llvm.ptr). We need to wrap it into a TVMFFIAny struct before
            // storing to retPtr.
            if (mlir::isa<mlir::torch::Torch::BaseTensorType>(operandTy)) {
              mlir::Value handle = result.value();
              mlir::Value wrapped =
                  mlir::LLVM::UndefOp::create(rewriter, loc, anyTy);
              mlir::Value typeIdx = mlir::LLVM::ConstantOp::create(
                  rewriter, loc, i32Ty, kTVMFFITensor);
              wrapped = mlir::LLVM::InsertValueOp::create(
                  rewriter, loc, anyTy, wrapped, typeIdx,
                  llvm::ArrayRef<int64_t>{0});
              mlir::Value vObj = mlir::LLVM::PtrToIntOp::create(
                  rewriter, loc, rewriter.getIntegerType(64), handle);
              wrapped = mlir::LLVM::InsertValueOp::create(rewriter, loc, anyTy,
                                                          wrapped, vObj, {2});
              mlir::LLVM::StoreOp::create(rewriter, loc, wrapped, retPtr);
            } else {
              // POD handlers return a full TVMFFIAny struct; store directly.
              mlir::LLVM::StoreOp::create(rewriter, loc, result.value(),
                                          retPtr);
            }
          }
          mlir::LLVM::ConstantOp cnst =
              mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, 0);
          mlir::LLVM::ReturnOp::create(rewriter, loc, cnst.getResult());
        } else {
          rewriter.clone(operation, mapping);
        }
      }
    }

    rewriter.replaceOp(op, llvmFunc);
    return mlir::success();
  }
};

class ConvertTVMFFIToLLVMPass
    : public impl::ConvertTVMFFIToLLVMBase<ConvertTVMFFIToLLVMPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::ConversionTarget target(context);
    mlir::LLVMTypeConverter typeConverter(&context);
    mlir::RewritePatternSet patterns(&context);

    libtriton::torch::setupBackendTypeConversion(target, typeConverter);
    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp>();
    populateTVMFFIToLLVMConversionPatterns(target, typeConverter, patterns);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct TVMFFIToLLVMDialectInterface
    : public mlir::ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(
      mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
      mlir::RewritePatternSet &patterns) const final {
    libtriton::torch::setupBackendTypeConversion(target, typeConverter);
    populateTVMFFIToLLVMConversionPatterns(target, typeConverter, patterns);
  }
};

} // namespace

void populateTVMFFIToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertFuncOp>(typeConverter, patterns.getContext());
  target.addIllegalDialect<TVMFFIDialect>();
  target.addLegalDialect<mlir::BuiltinDialect, mlir::func::FuncDialect,
                         mlir::LLVM::LLVMDialect,
                         mlir::torch::Torch::TorchDialect>();
}

void registerConvertTVMFFIToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, libtriton::tvm_ffi::TVMFFIDialect *dialect) {
        dialect->addInterfaces<TVMFFIToLLVMDialectInterface>();
      });
}

} // namespace libtriton::tvm_ffi
