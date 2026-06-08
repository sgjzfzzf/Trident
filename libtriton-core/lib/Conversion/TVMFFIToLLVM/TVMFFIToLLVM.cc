//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "libtriton-core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "libtriton-core/Conversion/TVMFFIToLLVM/TVMFFICAPIDescriptors.h"
#include "libtriton-core/Conversion/Utils/IConstantUtils.h"
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

static mlir::LLVM::LLVMStructType getTVMFFIAnyType(mlir::MLIRContext *context) {
  mlir::Type i32Ty = mlir::IntegerType::get(context, 32);
  mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
  return mlir::LLVM::LLVMStructType::getLiteral(context, {i32Ty, i32Ty, i64Ty},
                                                true);
}

/// Given a void* pointing to a TVMFFIAny ({i32, i32, i64}) struct in an array,
/// return a pointer to its i64 payload field (field index 2).
static mlir::Value getTVMFFIAnyPayloadPtr(mlir::OpBuilder &rewriter,
                                          mlir::Value slotPtr) {
  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(ctx);
  return mlir::LLVM::GEPOp::create(rewriter, slotPtr.getLoc(), ptrTy, anyTy,
                                   slotPtr,
                                   mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
}

//===----------------------------------------------------------------------===//
// Type conversion handlers for packing/unpacking TVM FFI arguments
//===----------------------------------------------------------------------===//

/// CRTP base: auto-generates matches(mlir::Type) from the Torch type parameter.
template <typename TorchType> struct TypeHandlerBase {
  static bool matches(mlir::Type type) { return mlir::isa<TorchType>(type); }
};

struct BaseTensorHandler : TypeHandlerBase<mlir::torch::Torch::BaseTensorType> {
  /// Pack an AtenTensorHandle (input) into a TVMFFIAny slot (ptr).
  ///
  /// Flow:
  /// 1. Extract tensor properties via aoti_torch_get_* functions.
  /// 2. Reverse-map Torch dtype/device → DLPack dtype/device.
  /// 3. Heap-allocate a DLManagedTensor, fill with extracted properties.
  /// 4. Convert to TVMFFIObjectHandle via TVMFFITensorFromDLPack.
  /// 5. Store type_index=kTVMFFITensor(70) and payload=handle in TVMFFIAny.
  static mlir::LogicalResult store(mlir::OpBuilder &builder, mlir::Value input,
                                   mlir::Value ptr) {
    using namespace libtriton::tvm_ffi::capi;
    mlir::Location loc = input.getLoc();
    mlir::MLIRContext *context = builder.getContext();

    mlir::ModuleOp moduleOp =
        input.getDefiningOp()->getParentOfType<mlir::ModuleOp>();

    // Declare the runtime pack function.
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> packFn =
        getOrCreatePackTensorToTVMFFIAny(moduleOp);
    if (mlir::failed(packFn)) {
      return mlir::failure();
    }

    // Call mLibTritonPackTensorToTVMFFIAny(input, ptr).
    // The runtime function handles all tensor property extraction,
    // dtype/device mapping, DLManagedTensor creation, and TVMFFIAny
    // population. Returns 0 on success, non-zero on failure.
    mlir::LLVM::CallOp::create(builder, loc, *packFn,
                               mlir::ValueRange{input, ptr});

    return mlir::success();
  }
  static mlir::FailureOr<mlir::Value> load(mlir::OpBuilder &builder,
                                           mlir::Value ptr) {
    mlir::Location loc = ptr.getLoc();
    mlir::MLIRContext *context = builder.getContext();
    mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);

    mlir::ModuleOp moduleOp =
        ptr.getDefiningOp()->getParentOfType<mlir::ModuleOp>();

    // Declare the runtime unpack function.
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> unpackFn =
        capi::getOrCreateUnpackTVMFFIAnyToTensor(moduleOp);
    if (mlir::failed(unpackFn)) {
      return mlir::failure();
    }

    // Allocate slot for output AtenTensorHandle.
    mlir::Value handleSlot = mlir::LLVM::AllocaOp::create(
        builder, loc, ptrTy, ptrTy,
        conversion::utils::emitI64Constant(builder, loc, context, 1));

    // Call mLibTritonUnpackTVMFFIAnyToTensor(ptr, &handleSlot).
    // The runtime function handles DLTensor extraction (with object offset),
    // dtype/device mapping, and aoti_torch_create_tensor_from_blob.
    // Returns 0 on success, non-zero on failure.
    mlir::LLVM::CallOp::create(builder, loc, *unpackFn,
                               mlir::ValueRange{ptr, handleSlot});

    return mlir::LLVM::LoadOp::create(builder, loc, ptrTy, handleSlot)
        .getResult();
  }
};

struct BoolHandler : TypeHandlerBase<mlir::torch::Torch::BoolType> {
  static mlir::LogicalResult store(mlir::OpBuilder &builder, mlir::Value input,
                                   mlir::Value ptr) {
    mlir::Location loc = input.getLoc();
    mlir::Value payloadPtr = getTVMFFIAnyPayloadPtr(builder, ptr);
    mlir::Value ext = mlir::LLVM::ZExtOp::create(
        builder, loc, mlir::IntegerType::get(builder.getContext(), 64), input);
    mlir::LLVM::StoreOp::create(builder, loc, ext, payloadPtr);
    return mlir::success();
  }
  static mlir::FailureOr<mlir::Value> load(mlir::OpBuilder &builder,
                                           mlir::Value ptr) {
    mlir::Location loc = ptr.getLoc();
    mlir::Value payload = getTVMFFIAnyPayloadPtr(builder, ptr);
    mlir::Value loaded = mlir::LLVM::LoadOp::create(
        builder, loc, mlir::IntegerType::get(builder.getContext(), 64),
        payload);
    return mlir::LLVM::TruncOp::create(
               builder, loaded.getLoc(),
               mlir::IntegerType::get(builder.getContext(), 1), loaded)
        .getResult();
  }
};

struct IntHandler : TypeHandlerBase<mlir::torch::Torch::IntType> {
  static mlir::LogicalResult store(mlir::OpBuilder &builder, mlir::Value input,
                                   mlir::Value ptr) {
    mlir::Value payload = getTVMFFIAnyPayloadPtr(builder, ptr);
    mlir::LLVM::StoreOp::create(builder, input.getLoc(), input, payload);
    return mlir::success();
  }
  static mlir::FailureOr<mlir::Value> load(mlir::OpBuilder &builder,
                                           mlir::Value ptr) {
    mlir::Value payload = getTVMFFIAnyPayloadPtr(builder, ptr);
    return mlir::LLVM::LoadOp::create(
               builder, ptr.getLoc(),
               mlir::IntegerType::get(builder.getContext(), 64), payload)
        .getResult();
  }
};

struct FloatHandler : TypeHandlerBase<mlir::torch::Torch::FloatType> {
  static mlir::LogicalResult store(mlir::OpBuilder &builder, mlir::Value input,
                                   mlir::Value ptr) {
    mlir::Location loc = input.getLoc();
    mlir::Value payload = getTVMFFIAnyPayloadPtr(builder, ptr);
    mlir::Value bits = mlir::LLVM::BitcastOp::create(
        builder, loc, mlir::IntegerType::get(builder.getContext(), 64), input);
    mlir::LLVM::StoreOp::create(builder, loc, bits, payload);
    return mlir::success();
  }
  static mlir::FailureOr<mlir::Value> load(mlir::OpBuilder &builder,
                                           mlir::Value ptr) {
    mlir::Location loc = ptr.getLoc();
    mlir::Value payload = getTVMFFIAnyPayloadPtr(builder, ptr);
    mlir::Value loaded = mlir::LLVM::LoadOp::create(
        builder, loc, mlir::IntegerType::get(builder.getContext(), 64),
        payload);
    return mlir::LLVM::BitcastOp::create(
               builder, loc, mlir::Float64Type::get(builder.getContext()),
               loaded)
        .getResult();
  }
};

struct NoneHandler : TypeHandlerBase<mlir::torch::Torch::NoneType> {
  static mlir::LogicalResult store(mlir::OpBuilder &builder, mlir::Value,
                                   mlir::Value ptr) {
    mlir::Location loc = ptr.getLoc();
    mlir::Value payload = getTVMFFIAnyPayloadPtr(builder, ptr);
    mlir::LLVM::StoreOp::create(builder, loc,
                                conversion::utils::emitI64Constant(
                                    builder, loc, builder.getContext(), 0),
                                payload);
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

template <typename... Handlers> struct TypeDispatch {
  static mlir::LogicalResult store(mlir::Type type, mlir::OpBuilder &builder,
                                   mlir::Value input, mlir::Value ptr) {
    mlir::LogicalResult result = mlir::failure();
    auto tryHandler = [&](auto handler) {
      if (mlir::failed(result) && decltype(handler)::matches(type)) {
        result = decltype(handler)::store(builder, input, ptr);
      }
    };
    (tryHandler(Handlers{}), ...);
    return result;
  }

  static mlir::FailureOr<mlir::Value>
  load(mlir::Type type, mlir::OpBuilder &builder, mlir::Value ptr) {
    mlir::FailureOr<mlir::Value> result = mlir::failure();
    auto tryHandler = [&](auto handler) {
      if (mlir::failed(result) && decltype(handler)::matches(type)) {
        result = decltype(handler)::load(builder, ptr);
      }
    };
    (tryHandler(Handlers{}), ...);
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

    // Create all blocks first: one per original block.
    for (mlir::Block &blk : op.getBody()) {
      mlir::Block *newBlock = rewriter.createBlock(&region);
      mapping.map(&blk, newBlock);
    }

    // The first block is the entry; add ABI arguments to it.
    // TVM-FFI C ABI: int32_t(void* handle, TVMFFIAny* args, int32_t num_args,
    // TVMFFIAny* result)
    mlir::Block *entryBlock = &region.front();
    entryBlock->addArgument(ptrTy, loc); // handle (unused)
    mlir::Value argsPtr = entryBlock->addArgument(ptrTy, loc); // args
    entryBlock->addArgument(i32Ty, loc); // num_args (unused)
    mlir::Value retPtr = entryBlock->addArgument(ptrTy, loc); // result

    // Load input payloads via AllHandlers::load, cast to Torch, map to
    // original block args.
    rewriter.setInsertionPointToStart(entryBlock);
    mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(context);
    mlir::Block &entry = op.getBody().front();
    for (auto [i, arg] : llvm::enumerate(entry.getArguments())) {
      mlir::Value slot =
          mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, anyTy, argsPtr,
                                    mlir::ArrayRef<mlir::LLVM::GEPArg>{i});
      mlir::Type argTy = arg.getType();
      mlir::FailureOr<mlir::Value> loaded =
          AllHandlers::load(argTy, rewriter, slot);
      if (mlir::failed(loaded)) {
        return op.emitError("unsupported input type: ") << argTy;
      }
      mlir::Value casted = mlir::UnrealizedConversionCastOp::create(
                               rewriter, loc, argTy, *loaded)
                               .getResult(0);
      mapping.map(arg, casted);
    }

    // Clone every op from every block into its corresponding new block.

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
            if (mlir::failed(
                    AllHandlers::store(operandTy, rewriter, casted, retPtr))) {
              return op.emitError("unsupported return type: ") << operandTy;
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
