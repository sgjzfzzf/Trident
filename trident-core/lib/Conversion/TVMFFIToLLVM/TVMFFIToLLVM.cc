//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "trident-core/Conversion/Utils/Check.h"
#include "trident-core/Conversion/Utils/GlobalString.h"
#include "trident-core/Conversion/Utils/TVMFFICAPIDescriptors.h"
#include "trident-core/Conversion/Utils/TVMFFIUtils.h"
#include "trident-core/Conversion/Utils/Type.h"
#include "trident-core/Dialect/TVMFFI/IR/TVMFFIAttributes.h"
#include "trident-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "trident-core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident-core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"
#include "llvm/Support/FormatVariadic.h"

#include <string>

namespace trident::tvm_ffi {

#define GEN_PASS_DEF_CONVERTTVMFFITOLLVM
#include "trident-core/Conversion/Passes.h.inc"

namespace {

/// Helper: given a TVMFFIAny* slot, load the TVMFFIObjectHandle from slot[2],
/// inttoptr it, and advance past the 24-byte header to produce a DLTensor*.
static mlir::Value getDLTensorPtr(mlir::OpBuilder &builder, mlir::Value slot) {
  mlir::Location loc = slot.getLoc();
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::LLVM::LLVMStructType anyTy =
      trident::conversion::utils::getTVMFFIAnyType(ctx);
  mlir::IntegerType i8Ty = mlir::IntegerType::get(ctx, 8);

  mlir::Value handlePtr =
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, slot,
                                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
  mlir::Value handleInt =
      mlir::LLVM::LoadOp::create(builder, loc, i64Ty, handlePtr);
  mlir::Value handleObj =
      mlir::LLVM::IntToPtrOp::create(builder, loc, ptrTy, handleInt);

  return mlir::LLVM::GEPOp::create(
      builder, loc, ptrTy, i8Ty, handleObj,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{sizeof(TVMFFIObject)});
}

//===----------------------------------------------------------------------===//
// Guard checks — dispatched on tvm_ffi.guard argument attributes
//===----------------------------------------------------------------------===//

static mlir::Value buildGuards(mlir::OpBuilder &builder, mlir::Value slot,
                               mlir::Attribute attr) {
  mlir::Location loc = slot.getLoc();
  mlir::MLIRContext *ctx = builder.getContext();

  mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::LLVM::LLVMStructType dlTensorTy =
      conversion::utils::getDLTensorType(ctx);
  mlir::LLVM::LLVMStructType anyTy =
      trident::conversion::utils::getTVMFFIAnyType(ctx);

  if (ConstantGuardAttr constantGuard =
          mlir::dyn_cast<ConstantGuardAttr>(attr)) {
    const int64_t expectedTypeIndex = constantGuard.getTypeIndex();
    const int64_t expectedPayload = constantGuard.getPayload();

    mlir::Value typeIndexPtr =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, slot,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0});
    mlir::Value loadedTypeIndex =
        mlir::LLVM::LoadOp::create(builder, loc, i32Ty, typeIndexPtr);

    mlir::Value payloadPtr =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, slot,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    mlir::Value loadedPayload =
        mlir::LLVM::LoadOp::create(builder, loc, i64Ty, payloadPtr);

    mlir::Value expectedTypeIndexVal =
        mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, expectedTypeIndex);
    mlir::Value expectedPayloadVal =
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, expectedPayload);

    mlir::Value typeCmp =
        mlir::LLVM::ICmpOp::create(builder, loc, mlir::LLVM::ICmpPredicate::eq,
                                   loadedTypeIndex, expectedTypeIndexVal);
    mlir::Value payloadCmp =
        mlir::LLVM::ICmpOp::create(builder, loc, mlir::LLVM::ICmpPredicate::eq,
                                   loadedPayload, expectedPayloadVal);
    return mlir::LLVM::AndOp::create(builder, loc, typeCmp, payloadCmp);
  } else if (CudaDeviceGuardAttr cudaGuard =
                 mlir::dyn_cast<tvm_ffi::CudaDeviceGuardAttr>(attr)) {
    const int64_t deviceType = cudaGuard.getDeviceType();
    const int64_t deviceIndex = cudaGuard.getDeviceIndex();

    mlir::Value dlTensorPtr = getDLTensorPtr(builder, slot);

    mlir::Value typeGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlTensorTy, dlTensorPtr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1, 0});
    mlir::Value loadedType =
        mlir::LLVM::LoadOp::create(builder, loc, i32Ty, typeGep);

    mlir::Value idGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlTensorTy, dlTensorPtr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1, 1});
    mlir::Value loadedId =
        mlir::LLVM::LoadOp::create(builder, loc, i32Ty, idGep);

    mlir::Value expectedType =
        mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, deviceType);
    mlir::Value typeCmp = mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, loadedType, expectedType);

    mlir::Value expectedId =
        mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, deviceIndex);
    mlir::Value idCmp = mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, loadedId, expectedId);

    return mlir::LLVM::AndOp::create(builder, loc, typeCmp, idCmp);
  } else if (DimensionGuardAttr dimGuard =
                 mlir::dyn_cast<DimensionGuardAttr>(attr)) {
    const int64_t expectedVal = dimGuard.getExpected();

    mlir::Value dlTensorPtr = getDLTensorPtr(builder, slot);

    mlir::Value ndimGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlTensorTy, dlTensorPtr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    mlir::Value ndimVal =
        mlir::LLVM::LoadOp::create(builder, loc, i32Ty, ndimGep);

    mlir::Value expected =
        mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, expectedVal);
    return mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, ndimVal, expected);
  } else if (DtypeGuardAttr dtypeGuard = mlir::dyn_cast<DtypeGuardAttr>(attr)) {
    const int64_t expectedCode = dtypeGuard.getCode();
    const int64_t expectedBits = dtypeGuard.getBits();
    const int64_t expectedLanes = dtypeGuard.getLanes();

    mlir::Value dlTensorPtr = getDLTensorPtr(builder, slot);

    mlir::Value dtypeGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlTensorTy, dlTensorPtr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 3});
    mlir::LLVM::LLVMStructType dlDtypeTy =
        conversion::utils::getDLDataType(ctx);

    mlir::IntegerType i8Ty = mlir::IntegerType::get(ctx, 8);
    mlir::IntegerType i16Ty = mlir::IntegerType::get(ctx, 16);

    mlir::Value codePtr =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlDtypeTy, dtypeGep,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0});
    mlir::Value bitsPtr =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlDtypeTy, dtypeGep,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1});
    mlir::Value lanesPtr =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlDtypeTy, dtypeGep,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2});

    mlir::Value loadedCode =
        mlir::LLVM::LoadOp::create(builder, loc, i8Ty, codePtr);
    mlir::Value loadedBits =
        mlir::LLVM::LoadOp::create(builder, loc, i8Ty, bitsPtr);
    mlir::Value loadedLanes =
        mlir::LLVM::LoadOp::create(builder, loc, i16Ty, lanesPtr);

    mlir::Value expectedCodeValue =
        mlir::LLVM::ConstantOp::create(builder, loc, i8Ty, expectedCode);
    mlir::Value expectedBitsValue =
        mlir::LLVM::ConstantOp::create(builder, loc, i8Ty, expectedBits);
    mlir::Value expectedLanesValue =
        mlir::LLVM::ConstantOp::create(builder, loc, i16Ty, expectedLanes);

    mlir::Value codeCmp =
        mlir::LLVM::ICmpOp::create(builder, loc, mlir::LLVM::ICmpPredicate::eq,
                                   loadedCode, expectedCodeValue);
    mlir::Value bitsCmp =
        mlir::LLVM::ICmpOp::create(builder, loc, mlir::LLVM::ICmpPredicate::eq,
                                   loadedBits, expectedBitsValue);
    mlir::Value lanesCmp =
        mlir::LLVM::ICmpOp::create(builder, loc, mlir::LLVM::ICmpPredicate::eq,
                                   loadedLanes, expectedLanesValue);

    mlir::Value codeAndBits =
        mlir::LLVM::AndOp::create(builder, loc, codeCmp, bitsCmp);
    return mlir::LLVM::AndOp::create(builder, loc, codeAndBits, lanesCmp);
  } else if (SizeGuardAttr sizeGuard = mlir::dyn_cast<SizeGuardAttr>(attr)) {
    const int64_t index = sizeGuard.getIndex();
    const int64_t expectedVal = sizeGuard.getExpected();

    mlir::Value dlTensorPtr = getDLTensorPtr(builder, slot);

    mlir::Value shapePtrGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlTensorTy, dlTensorPtr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 4});
    mlir::Value shapePtr =
        mlir::LLVM::LoadOp::create(builder, loc, ptrTy, shapePtrGep);

    mlir::Value elemGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, i64Ty, shapePtr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{index});
    mlir::Value elemVal =
        mlir::LLVM::LoadOp::create(builder, loc, i64Ty, elemGep);

    mlir::Value expected =
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, expectedVal);
    return mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, elemVal, expected);
  } else if (StorageOffsetGuardAttr offsetGuard =
                 mlir::dyn_cast<StorageOffsetGuardAttr>(attr)) {
    const int64_t expectedVal = offsetGuard.getExpected();

    mlir::Value dlTensorPtr = getDLTensorPtr(builder, slot);

    mlir::Value offsetGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlTensorTy, dlTensorPtr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 6});
    mlir::Value offsetVal =
        mlir::LLVM::LoadOp::create(builder, loc, i64Ty, offsetGep);

    mlir::Value expected =
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, expectedVal);
    return mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, offsetVal, expected);
  } else if (StrideGuardAttr strideGuard =
                 mlir::dyn_cast<StrideGuardAttr>(attr)) {
    const int64_t index = strideGuard.getIndex();
    const int64_t expectedVal = strideGuard.getExpected();

    mlir::Value dlTensorPtr = getDLTensorPtr(builder, slot);

    mlir::Value stridePtrGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlTensorTy, dlTensorPtr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 5});
    mlir::Value stridePtr =
        mlir::LLVM::LoadOp::create(builder, loc, ptrTy, stridePtrGep);

    mlir::Value elemGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, i64Ty, stridePtr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{index});
    mlir::Value elemVal =
        mlir::LLVM::LoadOp::create(builder, loc, i64Ty, elemGep);

    mlir::Value expected =
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, expectedVal);
    return mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, elemVal, expected);
  } else if (mlir::isa<TensorTypeGuardAttr>(attr)) {
    mlir::Value typeCodePtr =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, slot,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0});
    mlir::Value loadedTypeCode =
        mlir::LLVM::LoadOp::create(builder, loc, i32Ty, typeCodePtr);
    mlir::Value expected =
        mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, kTVMFFITensor);
    return mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, loadedTypeCode, expected);
  }
  return {};
}

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
    mlir::LLVM::LLVMStructType anyTy =
        trident::conversion::utils::getTVMFFIAnyType(context);
    for (auto [i, arg] : llvm::enumerate(op.getArguments())) {
      mlir::Value slot =
          mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, anyTy, argsPtr,
                                    llvm::ArrayRef<mlir::LLVM::GEPArg>{i});
      mlir::Value anySlot =
          mlir::LLVM::LoadOp::create(rewriter, loc, anyTy, slot);
      // All torch types uniformly lower to TVMFFIAny; pass through directly.
      mlir::Value casted = mlir::UnrealizedConversionCastOp::create(
                               rewriter, loc, arg.getType(), anySlot)
                               .getResult(0);
      mapping.map(arg, casted);

      // ── Guard checking: if any guard fails, return immediately with -1 ──
      if (mlir::ArrayAttr guardAttrs = mlir::dyn_cast_or_null<mlir::ArrayAttr>(
              op.getArgAttr(i, "tvm_ffi.guard"))) {
        for (mlir::Attribute g : guardAttrs) {
          mlir::Value guardResult = buildGuards(rewriter, slot, g);
          if (!guardResult) {
            return op.emitError("unsupported guard attribute on argument ")
                   << i;
          }
          mlir::Block *currentBlock = rewriter.getInsertionBlock();
          mlir::Block *failBlock = rewriter.createBlock(&region);
          rewriter.setInsertionPointToStart(failBlock);
          // Write an Exception object into retPtr so that the dispatcher
          // can identify this as a guard mismatch and try the next
          // specialization (or convert to Error if none left).
          mlir::ModuleOp moduleOp =
              op->template getParentOfType<mlir::ModuleOp>();
          if (!moduleOp) {
            return op.emitError("failed to get parent ModuleOp for guard "
                                "failure error reporting");
          }
          mlir::Value kindPtr = conversion::utils::getOrCreateGlobalString(
              rewriter, loc, moduleOp, "GuardMatchKind", "GuardMatch");

          // Build kTVMFFIRawStr (type_index = 8) TVMFFIAny for kind.
          mlir::IntegerType i64Ty = rewriter.getIntegerType(64);
          mlir::Value rawStrTypeIndex =
              mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, 8);
          auto buildRawStrAny = [&](mlir::Value strPtr) -> mlir::Value {
            mlir::Value any = mlir::LLVM::UndefOp::create(rewriter, loc, anyTy);
            any = mlir::LLVM::InsertValueOp::create(rewriter, loc, any,
                                                    rawStrTypeIndex,
                                                    llvm::ArrayRef<int64_t>{0});
            mlir::Value ptrAsI64 =
                mlir::LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, strPtr);
            any = mlir::LLVM::InsertValueOp::create(
                rewriter, loc, any, ptrAsI64, llvm::ArrayRef<int64_t>{2});
            return any;
          };
          mlir::Value one =
              mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1);

          mlir::Value kindSlot =
              mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, anyTy, one);
          mlir::LLVM::StoreOp::create(rewriter, loc, buildRawStrAny(kindPtr),
                                      kindSlot);

          // Call trident.ffi.Exception(kind) — only passes the kind argument.
          mlir::Value resultSlot = TRIDENT_CHECK(
              conversion::utils::callTVMFFIGlobalFunction(
                  rewriter, loc, moduleOp, "trident.ffi.Exception",
                  llvm::ArrayRef<mlir::Value>{kindSlot}),
              return op.emitError("failed to call trident.ffi.Exception"));
          mlir::Value exceptionAny =
              mlir::LLVM::LoadOp::create(rewriter, loc, anyTy, resultSlot);

          // Store the Exception into retPtr and return 0 (success).
          mlir::LLVM::StoreOp::create(rewriter, loc, exceptionAny, retPtr);
          mlir::LLVM::ConstantOp zeroReturn =
              mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, 0);
          mlir::LLVM::ReturnOp::create(rewriter, loc, zeroReturn);

          mlir::Block *contBlock = rewriter.createBlock(&region);
          rewriter.setInsertionPointToEnd(currentBlock);
          mlir::LLVM::CondBrOp::create(rewriter, loc, guardResult, contBlock,
                                       failBlock);

          rewriter.setInsertionPointToStart(contBlock);
        }
      }
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

    // Step 4: Branch from the last guard-check block to the first body block.
    assert(firstBodyBlock && "expected at least one block in function body");
    mlir::LLVM::BrOp::create(rewriter, loc, firstBodyBlock);

    // Step 6: Clone original function body ops into the mapped body blocks.
    for (mlir::Block &blk : op.getBody()) {
      mlir::Block *dest = mapping.lookupOrDefault(&blk);
      rewriter.setInsertionPointToEnd(dest);
      for (mlir::Operation &operation : llvm::make_early_inc_range(blk)) {
        if (ReturnOp returnOp = mlir::dyn_cast<ReturnOp>(&operation)) {
          for (mlir::Value operand : returnOp.getOperands()) {
            mlir::Value retVal = mapping.lookupOrDefault(operand);
            mlir::Type operandTy = operand.getType();
            // All torch types uniformly lower to TVMFFIAny; store directly.
            mlir::Value casted =
                mlir::UnrealizedConversionCastOp::create(
                    rewriter, loc, getTypeConverter()->convertType(operandTy),
                    retVal)
                    .getResult(0);
            mlir::LLVM::StoreOp::create(rewriter, loc, casted, retPtr);
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

    torch::setupBackendTypeConversion(target, typeConverter);
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
    torch::setupBackendTypeConversion(target, typeConverter);
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
      +[](mlir::MLIRContext *ctx, tvm_ffi::TVMFFIDialect *dialect) {
        dialect->addInterfaces<TVMFFIToLLVMDialectInterface>();
      });
}

} // namespace trident::tvm_ffi
