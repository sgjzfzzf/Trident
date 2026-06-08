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

/// Given a void* pointing to a TVMFFIAny ({i32, i32, i64}) struct in an array,
/// return a pointer to its i32 type_index field (field index 0).
static mlir::Value getTVMFFIAnyTypeIndexPtr(mlir::OpBuilder &rewriter,
                                            mlir::Value slotPtr) {
  mlir::MLIRContext *ctx = rewriter.getContext();
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(ctx);
  return mlir::LLVM::GEPOp::create(rewriter, slotPtr.getLoc(), ptrTy, anyTy,
                                   slotPtr,
                                   mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 0});
}

/// Creates an LLVM global string constant and returns a pointer to it.
/// Skips creation if an identically-named global already exists.
static mlir::Value createGlobalString(mlir::OpBuilder &builder,
                                      mlir::Location loc,
                                      mlir::ModuleOp moduleOp,
                                      llvm::StringRef name,
                                      llvm::StringRef content) {
  mlir::MLIRContext *context = moduleOp.getContext();
  const mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);
  const mlir::Type i8Ty = mlir::IntegerType::get(context, 8);
  std::string globalSymName =
      llvm::formatv("__tvm_ffi_err_{0}_{1}", name, content);
  if (!moduleOp.lookupSymbol<mlir::LLVM::GlobalOp>(globalSymName)) {
    const mlir::LLVM::LLVMArrayType arrayType =
        mlir::LLVM::LLVMArrayType::get(i8Ty, content.size());
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());
    mlir::LLVM::GlobalOp::create(builder, loc, arrayType, /*isConstant=*/true,
                                 mlir::LLVM::linkage::Linkage::Internal,
                                 globalSymName,
                                 /*value=*/builder.getStringAttr(content));
  }
  return mlir::LLVM::AddressOfOp::create(builder, loc, ptrTy, globalSymName)
      .getResult();
}

//===----------------------------------------------------------------------===//
// Type conversion handlers for packing/unpacking TVM FFI arguments
//===----------------------------------------------------------------------===//

/// CRTP base: auto-generates matches(mlir::Type) from the Torch type parameter.
template <typename TorchType> struct TypeHandlerBase {
  static bool matches(mlir::Type type) { return mlir::isa<TorchType>(type); }
};

/// CRTP base extended for POD type handlers that need a single TypeIndex check.
/// ExpectedTypeIndex is the TVMFFITypeIndex constant to validate at runtime.
template <typename TorchType, int32_t ExpectedTypeIndex>
struct PodTypeHandlerBase : TypeHandlerBase<TorchType> {
  /// Emit runtime type_index check for the expected value.
  /// Splits block and creates error path on mismatch.
  /// Returns the continueBlock, or nullptr on failure.
  static mlir::Block *check(mlir::OpBuilder &builder, mlir::Value ptr) {
    mlir::Location loc = ptr.getLoc();
    mlir::MLIRContext *context = builder.getContext();
    mlir::Type i32Ty = mlir::IntegerType::get(context, 32);

    mlir::Value typeIndexPtr = getTVMFFIAnyTypeIndexPtr(builder, ptr);
    mlir::Value typeIndex =
        mlir::LLVM::LoadOp::create(builder, loc, i32Ty, typeIndexPtr);
    mlir::Value expected = conversion::utils::emitI32Constant(
        builder, loc, context, ExpectedTypeIndex);
    mlir::Value mismatch = mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::ne, typeIndex, expected);

    mlir::Block *currentBlock = builder.getInsertionBlock();
    mlir::Block *continueBlock =
        currentBlock->splitBlock(builder.getInsertionPoint());
    mlir::Block *errorBlock = builder.createBlock(continueBlock);

    mlir::ModuleOp moduleOp =
        currentBlock->getParentOp()->getParentOfType<mlir::ModuleOp>();

    builder.setInsertionPointToEnd(currentBlock);
    mlir::LLVM::CondBrOp::create(builder, loc, mismatch, errorBlock,
                                 mlir::ValueRange{}, continueBlock,
                                 mlir::ValueRange{});

    builder.setInsertionPointToEnd(errorBlock);
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> errorFn =
        capi::getOrCreateTVMFFIErrorSetRaisedFromCStr(moduleOp);
    if (mlir::failed(errorFn)) {
      return nullptr;
    }
    mlir::Value kindStr =
        createGlobalString(builder, loc, moduleOp, "kind", "TypeError");
    mlir::Value msgStr = createGlobalString(builder, loc, moduleOp, "msg",
                                            "tvm_ffi: argument type mismatch");
    mlir::LLVM::CallOp::create(builder, loc, *errorFn,
                               mlir::ValueRange{kindStr, msgStr});
    mlir::Value minusOne =
        conversion::utils::emitI32Constant(builder, loc, context, -1);
    mlir::LLVM::ReturnOp::create(builder, loc, minusOne);

    return continueBlock;
  }
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
    mlir::Location loc = input.getLoc();
    mlir::MLIRContext *context = builder.getContext();

    mlir::ModuleOp moduleOp =
        input.getDefiningOp()->getParentOfType<mlir::ModuleOp>();

    // Declare the runtime pack function.
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> packFn =
        libtriton::tvm_ffi::capi::getOrCreatePackTensorToTVMFFIAny(moduleOp);
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
    mlir::Type i32Ty = mlir::IntegerType::get(context, 32);
    mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);

    mlir::Block *currentBlock = builder.getInsertionBlock();
    mlir::ModuleOp moduleOp =
        currentBlock->getParentOp()->getParentOfType<mlir::ModuleOp>();

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
    // Returns 0 on success, -1 on type mismatch (the runtime already checks
    // for kTVMFFITensor or kTVMFFIDLTensorPtr).
    mlir::Value ret =
        mlir::LLVM::CallOp::create(builder, loc, *unpackFn,
                                   mlir::ValueRange{ptr, handleSlot})
            .getResult();

    // Check if return value is -1 (type mismatch).
    mlir::Value minusOne =
        conversion::utils::emitI32Constant(builder, loc, context, -1);
    mlir::Value mismatch = mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, ret, minusOne);

    // Split block for error path.
    mlir::Block *continueBlock =
        currentBlock->splitBlock(builder.getInsertionPoint());
    mlir::Block *errorBlock = builder.createBlock(continueBlock);

    // CondBrOp: mismatch -> errorBlock, !mismatch -> continueBlock.
    builder.setInsertionPointToEnd(currentBlock);
    mlir::LLVM::CondBrOp::create(builder, loc, mismatch, errorBlock,
                                 mlir::ValueRange{}, continueBlock,
                                 mlir::ValueRange{});

    // Error block: set runtime error and propagate -1.
    builder.setInsertionPointToEnd(errorBlock);
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> errorFn =
        capi::getOrCreateTVMFFIErrorSetRaisedFromCStr(moduleOp);
    if (mlir::failed(errorFn)) {
      return mlir::failure();
    }
    mlir::Value kindStr =
        createGlobalString(builder, loc, moduleOp, "kind", "TypeError");
    mlir::Value msgStr = createGlobalString(builder, loc, moduleOp, "msg",
                                            "tvm_ffi: argument type mismatch");
    mlir::LLVM::CallOp::create(builder, loc, *errorFn,
                               mlir::ValueRange{kindStr, msgStr});
    mlir::LLVM::ReturnOp::create(builder, loc, minusOne);

    // Continue block: return loaded handle.
    builder.setInsertionPointToStart(continueBlock);
    return mlir::LLVM::LoadOp::create(builder, loc, ptrTy, handleSlot)
        .getResult();
  }
};

struct BoolHandler
    : PodTypeHandlerBase<mlir::torch::Torch::BoolType, kTVMFFIBool> {
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
    mlir::Block *cb = check(builder, ptr);
    if (!cb) {
      return mlir::failure();
    }
    builder.setInsertionPointToStart(cb);
    mlir::Value payload = getTVMFFIAnyPayloadPtr(builder, ptr);
    mlir::Value loaded = mlir::LLVM::LoadOp::create(
        builder, ptr.getLoc(), mlir::IntegerType::get(builder.getContext(), 64),
        payload);
    return mlir::LLVM::TruncOp::create(
               builder, loaded.getLoc(),
               mlir::IntegerType::get(builder.getContext(), 1), loaded)
        .getResult();
  }
};

struct IntHandler
    : PodTypeHandlerBase<mlir::torch::Torch::IntType, kTVMFFIInt> {
  static mlir::LogicalResult store(mlir::OpBuilder &builder, mlir::Value input,
                                   mlir::Value ptr) {
    mlir::Value payload = getTVMFFIAnyPayloadPtr(builder, ptr);
    mlir::LLVM::StoreOp::create(builder, input.getLoc(), input, payload);
    return mlir::success();
  }
  static mlir::FailureOr<mlir::Value> load(mlir::OpBuilder &builder,
                                           mlir::Value ptr) {
    mlir::Block *cb = check(builder, ptr);
    if (!cb) {
      return mlir::failure();
    }
    builder.setInsertionPointToStart(cb);
    mlir::Value payload = getTVMFFIAnyPayloadPtr(builder, ptr);
    return mlir::LLVM::LoadOp::create(
               builder, ptr.getLoc(),
               mlir::IntegerType::get(builder.getContext(), 64), payload)
        .getResult();
  }
};

struct FloatHandler
    : PodTypeHandlerBase<mlir::torch::Torch::FloatType, kTVMFFIFloat> {
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
    mlir::Block *cb = check(builder, ptr);
    if (!cb) {
      return mlir::failure();
    }
    builder.setInsertionPointToStart(cb);
    mlir::Value payload = getTVMFFIAnyPayloadPtr(builder, ptr);
    mlir::Value loaded = mlir::LLVM::LoadOp::create(
        builder, ptr.getLoc(), mlir::IntegerType::get(builder.getContext(), 64),
        payload);
    return mlir::LLVM::BitcastOp::create(
               builder, loaded.getLoc(),
               mlir::Float64Type::get(builder.getContext()), loaded)
        .getResult();
  }
};

struct NoneHandler
    : PodTypeHandlerBase<mlir::torch::Torch::NoneType, kTVMFFINone> {
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
  static mlir::FailureOr<mlir::Value> load(mlir::OpBuilder &builder,
                                           mlir::Value ptr) {
    mlir::Block *cb = check(builder, ptr);
    if (!cb) {
      return mlir::failure();
    }
    builder.setInsertionPointToStart(cb);
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
