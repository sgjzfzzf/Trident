#include <cstddef>
#include <cstdint>
#include <optional>

#include "libtriton-core/Conversion/DLPackToLLVM/DLPackDTypeUtils.h"
#include "libtriton-core/Conversion/DLPackToLLVM/DLPackLLVMDescriptors.h"
#include "libtriton-core/Conversion/TVMFFIToLLVM/TVMFFICAPIDescriptors.h"
#include "libtriton-core/Conversion/TVMFFIToLLVM/TVMFFILLVMDescriptors.h"
#include "libtriton-core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "libtriton-core/Conversion/TVMFFIToLLVM/ToConvertPatterns.h"
#include "libtriton-core/Conversion/Utils/IConstantUtils.h"
#include "libtriton-core/Conversion/Utils/RuntimeCFunctionDeclUtils.h"
#include "libtriton-core/Conversion/Utils/StdLibCFunctionDeclUtils.h"
#include "libtriton-core/Dialect/DLPack/IR/DLPackDialect.h"
#include "libtriton-core/Dialect/DLPack/IR/DLPackOps.h"
#include "libtriton-core/Dialect/DLPack/IR/DLPackTypes.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tvm/ffi/c_api.h"
#include "tvm/ffi/extra/c_env_api.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"

namespace libtriton::tvm_ffi {

#define GEN_PASS_DEF_CONVERTTVMFFITOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

mlir::FailureOr<mlir::Value> emitEnvTensorAllocAsObjectHandle(
    mlir::ModuleOp moduleOp, mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Type dtype, llvm::ArrayRef<int64_t> shape) {
  mlir::MLIRContext *context = moduleOp.getContext();
  const mlir::Type i32Ty = mlir::IntegerType::get(context, 32);
  const mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
  const mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);
  const mlir::Type dlDataTypeTy =
      conversion::utils::DLDataTypeLLVMDescriptor::getLLVMType(context);
  const mlir::Type dlDeviceTy =
      conversion::utils::DLDeviceLLVMDescriptor::getLLVMType(context);
  const mlir::Type dlTensorTy =
      conversion::utils::DLTensorLLVMDescriptor::getLLVMType(context);

  const std::optional<DLDataType> dtypeInfo =
      conversion::utils::getDLPackDTypeFromMLIRType(dtype);
  if (!dtypeInfo.has_value()) {
    return mlir::failure();
  }

  const mlir::FailureOr<mlir::LLVM::LLVMFuncOp> currentDeviceOrErr =
      conversion::utils::runtime::getOrCreateGetCurrentDevice(moduleOp);
  if (mlir::failed(currentDeviceOrErr)) {
    return mlir::failure();
  }
  const mlir::FailureOr<mlir::LLVM::LLVMFuncOp> envAllocOrErr =
      conversion::utils::getOrCreateCAPI<decltype(&TVMFFIEnvTensorAlloc)>(
          moduleOp, "TVMFFIEnvTensorAlloc");
  if (mlir::failed(envAllocOrErr)) {
    return mlir::failure();
  }

  const mlir::Value shapeSlot =
      mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, i64Ty,
                                   conversion::utils::emitI64Constant(
                                       rewriter, loc, context, shape.size()))
          .getResult();
  for (size_t i = 0; i < shape.size(); ++i) {
    const mlir::Value dimPtr =
        mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, i64Ty, shapeSlot, {i})
            .getResult();
    mlir::LLVM::StoreOp::create(
        rewriter, loc,
        conversion::utils::emitI64Constant(rewriter, loc, context, shape[i]),
        dimPtr);
  }

  const mlir::TypedValue<mlir::IntegerType> dtypeCodeValue =
      conversion::utils::emitI8Constant(rewriter, loc, context,
                                        dtypeInfo->code);
  const mlir::TypedValue<mlir::IntegerType> dtypeBitsValue =
      conversion::utils::emitI8Constant(rewriter, loc, context,
                                        dtypeInfo->bits);
  const mlir::TypedValue<mlir::IntegerType> dtypeLanesValue =
      conversion::utils::emitI16Constant(rewriter, loc, context,
                                         dtypeInfo->lanes);
  const conversion::utils::DLDataTypeLLVMDescriptor dlDataTypeValue =
      conversion::utils::DLDataTypeLLVMDescriptor::build(
          rewriter, loc, mlir::cast<mlir::LLVM::LLVMStructType>(dlDataTypeTy),
          dtypeCodeValue, dtypeBitsValue, dtypeLanesValue);
  const mlir::TypedValue<mlir::IntegerType> ndimValue =
      conversion::utils::emitI32Constant(rewriter, loc, context, shape.size());
  mlir::LLVM::CallOp currentDeviceCall = mlir::LLVM::CallOp::create(
      rewriter, loc, *currentDeviceOrErr, mlir::ValueRange{});
  conversion::utils::DLDeviceLLVMDescriptor deviceValue =
      conversion::utils::DLDeviceLLVMDescriptor::from(
          currentDeviceCall.getResult());

  const mlir::TypedValue<mlir::LLVM::LLVMPointerType> nullPtrValue =
      mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
          mlir::LLVM::ZeroOp::create(rewriter, loc, ptrTy).getResult());
  const mlir::TypedValue<mlir::IntegerType> zeroOffsetValue =
      conversion::utils::emitI64Constant(rewriter, loc, context, 0);
  const mlir::TypedValue<mlir::LLVM::LLVMPointerType> shapeSlotValue =
      mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(shapeSlot);
  const conversion::utils::DLTensorLLVMDescriptor tensorValue =
      conversion::utils::DLTensorLLVMDescriptor::build(
          rewriter, loc, mlir::cast<mlir::LLVM::LLVMStructType>(dlTensorTy),
          nullPtrValue, deviceValue, ndimValue, dlDataTypeValue, shapeSlotValue,
          nullPtrValue, zeroOffsetValue);

  const mlir::TypedValue<mlir::IntegerType> oneValue =
      conversion::utils::emitI64Constant(rewriter, loc, context, 1);
  const mlir::TypedValue<mlir::LLVM::LLVMPointerType> tensorSlotValue =
      mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
          mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, dlTensorTy,
                                       oneValue)
              .getResult());
  mlir::LLVM::StoreOp::create(rewriter, loc, tensorValue.as(), tensorSlotValue);
  const mlir::TypedValue<mlir::LLVM::LLVMPointerType> outSlotValue =
      mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
          mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, ptrTy, oneValue)
              .getResult());
  mlir::LLVM::StoreOp::create(rewriter, loc, nullPtrValue, outSlotValue);

  mlir::LLVM::CallOp::create(rewriter, loc, *envAllocOrErr,
                             mlir::ValueRange{tensorSlotValue, outSlotValue});
  return mlir::LLVM::LoadOp::create(rewriter, loc, ptrTy, outSlotValue)
      .getResult();
}

struct LowerEnvTensorAllocOp
    : public mlir::OpConversionPattern<EnvTensorAllocOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(EnvTensorAllocOp op, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return mlir::failure();
    }
    mlir::Location loc = op.getLoc();
    mlir::FailureOr<mlir::Value> objectHandle =
        emitEnvTensorAllocAsObjectHandle(moduleOp, rewriter, loc, op.getDtype(),
                                         op.getShape());
    if (mlir::failed(objectHandle)) {
      return mlir::failure();
    } else {
      rewriter.replaceOp(op, *objectHandle);
      return mlir::success();
    }
  }
};

struct LowerToOp : public mlir::OpConversionPattern<ToOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ToOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const std::optional<mlir::Value> convertedValue =
        ToPatternSet::convert(op, adaptor, rewriter, getTypeConverter());
    if (!convertedValue.has_value()) {
      return mlir::failure();
    } else {
      rewriter.replaceOp(op, *convertedValue);
      return mlir::success();
    }
  }
};

struct LowerGetTypeIndexOp : public mlir::OpConversionPattern<GetTypeIndexOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(GetTypeIndexOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    const conversion::utils::TVMFFIAnyLLVMDescriptor anyValue =
        conversion::utils::TVMFFIAnyLLVMDescriptor::from(adaptor.getInput());
    rewriter.replaceOp(op, anyValue.typeIndex(rewriter, loc));
    return mlir::success();
  }
};

struct LowerErrorSetRaisedFromCStrOp
    : public mlir::OpConversionPattern<ErrorSetRaisedFromCStrOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ErrorSetRaisedFromCStrOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return mlir::failure();
    }
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> calleeOrErr =
        capi::getOrCreateTVMFFIErrorSetRaisedFromCStr(moduleOp);
    if (mlir::failed(calleeOrErr)) {
      return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, *calleeOrErr,
        mlir::ValueRange{adaptor.getKind(), adaptor.getMessage()});
    return mlir::success();
  }
};

struct LowerObjectIncRefOp : public mlir::OpConversionPattern<ObjectIncRefOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ObjectIncRefOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return mlir::failure();
    }
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> calleeOrErr =
        capi::getOrCreateTVMFFIObjectIncRef(moduleOp);
    if (mlir::failed(calleeOrErr)) {
      return mlir::failure();
    }
    mlir::LLVM::CallOp::create(rewriter, op->getLoc(), *calleeOrErr,
                               mlir::ValueRange{adaptor.getObject()});
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct LowerObjectDecRefOp : public mlir::OpConversionPattern<ObjectDecRefOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ObjectDecRefOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return mlir::failure();
    }
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> calleeOrErr =
        capi::getOrCreateTVMFFIObjectDecRef(moduleOp);
    if (mlir::failed(calleeOrErr)) {
      return mlir::failure();
    }
    // Create the call and erase the old op separately instead of using
    // replaceOpWithNewOp, because ObjectDecRefOp has 0 results but the
    // CallOp has 1 result (i32 return). The conversion framework does not
    // handle mismatched result counts correctly.
    mlir::LLVM::CallOp::create(rewriter, op->getLoc(), *calleeOrErr,
                               mlir::ValueRange{adaptor.getObject()});
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct LowerGetOpaquePtrOp : public mlir::OpConversionPattern<GetOpaquePtrOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(GetOpaquePtrOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();
    const mlir::Type i8Ty = mlir::IntegerType::get(context, 8);
    const mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);
    const mlir::Value cellPtr = mlir::LLVM::GEPOp::create(
        rewriter, loc, ptrTy, i8Ty, adaptor.getInput(),
        llvm::ArrayRef<mlir::LLVM::GEPArg>{sizeof(TVMFFIObject)});
    rewriter.replaceOp(op, cellPtr);
    return mlir::success();
  }
};

struct LowerLoadOp : public mlir::OpConversionPattern<LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    const mlir::Type convertedType =
        getTypeConverter()->convertType(op.getOutput().getType());
    if (!convertedType) {
      return mlir::failure();
    }
    const mlir::Value loaded = mlir::LLVM::LoadOp::create(
        rewriter, loc, convertedType, adaptor.getInput());
    rewriter.replaceOp(op, loaded);
    return mlir::success();
  }
};

struct LowerStoreOp : public mlir::OpConversionPattern<StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    const mlir::Value adaptedValue = adaptor.getValue();
    const mlir::Value adaptedPtr = adaptor.getPtr();
    const mlir::Type originalValueType = op.getValue().getType();

    // ObjectHandleType is just !llvm.ptr at the LLVM level — store the raw
    // pointer directly so that a subsequent tvm_ffi.load :
    // !tvm_ffi.object_handle (which also loads !llvm.ptr) is size-compatible.
    if (mlir::isa<ObjectHandleType>(originalValueType)) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, adaptedValue,
                                                       adaptedPtr);
      return mlir::success();
    }

    const std::optional<mlir::Value> anyStruct = toAnyStruct(
        adaptedValue, originalValueType, rewriter, loc, getTypeConverter());
    if (!anyStruct.has_value()) {
      return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, *anyStruct,
                                                     adaptedPtr);
    return mlir::success();
  }
};

struct LowerFunctionGetGlobalOp
    : public mlir::OpConversionPattern<FunctionGetGlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(FunctionGetGlobalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return mlir::failure();
    }
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();
    const mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);
    const mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
    const mlir::Type i8Ty = mlir::IntegerType::get(context, 8);

    llvm::StringRef funcName = op.getFuncName();
    const size_t nameLen = funcName.size();

    // Get or create TVMFFIFunctionGetGlobal function declaration.
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> calleeOrErr =
        capi::getOrCreateTVMFFIFunctionGetGlobal(moduleOp);
    if (mlir::failed(calleeOrErr)) {
      return mlir::failure();
    }

    // Create a global string constant for the function name at module level.
    const std::string globalSymName =
        llvm::formatv("__libtriton_tvm_ffi_func_name_{0}", funcName);
    const mlir::LLVM::LLVMArrayType arrayType =
        mlir::LLVM::LLVMArrayType::get(i8Ty, nameLen);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      mlir::LLVM::GlobalOp::create(
          rewriter, loc, arrayType, /*isConstant=*/true,
          mlir::LLVM::linkage::Linkage::Internal, globalSymName,
          /*value=*/rewriter.getStringAttr(funcName));
    }

    // Get the address of the global string.
    const mlir::Value addrOfOp =
        mlir::LLVM::AddressOfOp::create(rewriter, loc, ptrTy, globalSymName)
            .getResult();

    // String length in i64.
    const mlir::Value lenValue =
        conversion::utils::emitI64Constant(rewriter, loc, context, nameLen);

    // Alloca the TVMFFIByteArray struct = {ptr, i64}.
    const mlir::LLVM::LLVMStructType byteArrayStructTy =
        mlir::LLVM::LLVMStructType::getLiteral(context, {ptrTy, i64Ty});
    const mlir::Value oneValue =
        conversion::utils::emitI64Constant(rewriter, loc, context, 1);
    const mlir::Value byteArraySlot =
        mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, byteArrayStructTy,
                                     oneValue)
            .getResult();

    // Store data pointer (field 0).
    const mlir::Value dataPtrGep =
        mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, byteArrayStructTy,
                                  byteArraySlot,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0})
            .getResult();
    mlir::LLVM::StoreOp::create(rewriter, loc, addrOfOp, dataPtrGep);

    // Store size (field 1).
    const mlir::Value sizeGep =
        mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, byteArrayStructTy,
                                  byteArraySlot,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1})
            .getResult();
    mlir::LLVM::StoreOp::create(rewriter, loc, lenValue, sizeGep);

    // Alloca output slot (ptr = TVMFFIObjectHandle) initialized to null.
    const mlir::Value nullPtr =
        mlir::LLVM::ZeroOp::create(rewriter, loc, ptrTy).getResult();
    const mlir::Value outSlot =
        mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, ptrTy, oneValue)
            .getResult();
    mlir::LLVM::StoreOp::create(rewriter, loc, nullPtr, outSlot);

    // Call TVMFFIFunctionGetGlobal(bytearray_ptr, out_slot).
    mlir::LLVM::CallOp::create(rewriter, loc, *calleeOrErr,
                               mlir::ValueRange{byteArraySlot, outSlot});

    // Load the result.
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, ptrTy, outSlot);
    return mlir::success();
  }
};

struct LowerFunctionCallOp : public mlir::OpConversionPattern<FunctionCallOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(FunctionCallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return mlir::failure();
    }
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();
    const mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);
    const mlir::Type i32Ty = mlir::IntegerType::get(context, 32);
    const mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
    const mlir::Type anyStructTy =
        conversion::utils::TVMFFIAnyLLVMDescriptor::getLLVMType(context);

    // Get or create TVMFFIFunctionCall function declaration.
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> calleeOrErr =
        capi::getOrCreateTVMFFIFunctionCall(moduleOp);
    if (mlir::failed(calleeOrErr)) {
      return mlir::failure();
    }

    size_t numArgs = adaptor.getArgs().size();

    // Allocate the args array: TVMFFIAny[numArgs].
    const mlir::TypedValue<mlir::IntegerType> numArgsI64 =
        conversion::utils::emitI64Constant(rewriter, loc, context, numArgs);
    const mlir::Value argsSlot =
        mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, anyStructTy,
                                     numArgsI64)
            .getResult();

    // Store each argument into the args array.
    for (size_t i = 0; i < numArgs; ++i) {
      const mlir::Value argPtr =
          mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, anyStructTy, argsSlot,
                                    llvm::ArrayRef<mlir::LLVM::GEPArg>{i})
              .getResult();
      mlir::LLVM::StoreOp::create(rewriter, loc, adaptor.getArgs()[i], argPtr);
    }

    // Allocate the result slot.
    const mlir::TypedValue<mlir::IntegerType> oneValue =
        conversion::utils::emitI64Constant(rewriter, loc, context, 1);
    const mlir::Value resultSlot =
        mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, anyStructTy,
                                     oneValue)
            .getResult();

    // Initialize result slot to zero.
    const mlir::TypedValue<mlir::IntegerType> zeroI32 =
        conversion::utils::emitI32Constant(rewriter, loc, context, 0);
    const mlir::Value resultTypeIndexPtr =
        mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, anyStructTy, resultSlot,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0})
            .getResult();
    mlir::LLVM::StoreOp::create(rewriter, loc, zeroI32, resultTypeIndexPtr);

    // Call TVMFFIFunctionCall(func, args, num_args, &result).
    const mlir::TypedValue<mlir::IntegerType> numArgsI32 =
        conversion::utils::emitI32Constant(rewriter, loc, context, numArgs);
    mlir::LLVM::CallOp::create(
        rewriter, loc, *calleeOrErr,
        mlir::ValueRange{adaptor.getFunc(), argsSlot, numArgsI32, resultSlot});

    // Load the result.
    const mlir::Value result =
        mlir::LLVM::LoadOp::create(rewriter, loc, anyStructTy, resultSlot)
            .getResult();

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class ConvertTVMFFIToLLVMPass
    : public impl::ConvertTVMFFIToLLVMBase<ConvertTVMFFIToLLVMPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::LLVMTypeConverter typeConverter(&context);
    mlir::ConversionTarget target(context);
    mlir::RewritePatternSet patterns(&context);
    populateTVMFFIToLLVMConversionPatterns(target, typeConverter, patterns);
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, typeConverter);
    mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addIllegalDialect<TVMFFIDialect>();
    target.addLegalDialect<mlir::BuiltinDialect, mlir::LLVM::LLVMDialect>();
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                 typeConverter.isLegal(&op.getBody());
        });
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::Operation *op) {
          return mlir::isLegalForReturnOpTypeConversionPattern(op,
                                                               typeConverter);
        });
    target.markUnknownOpDynamicallyLegal([](mlir::Operation *op) {
      return !llvm::isa<TVMFFIDialect>(op->getDialect());
    });

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
    populateTVMFFIToLLVMConversionPatterns(target, typeConverter, patterns);
  }
};

void populateTVMFFIToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  mlir::MLIRContext *context = patterns.getContext();
  typeConverter.addConversion(
      [&](dlpack::DLManagedTensorType type) -> mlir::Type {
        return conversion::utils::DLManagedTensorLLVMDescriptor::getLLVMType(
            type.getContext());
      });
  typeConverter.addConversion([&](dlpack::DLTensorType type) -> mlir::Type {
    return conversion::utils::DLTensorLLVMDescriptor::getLLVMType(
        type.getContext());
  });
  typeConverter.addConversion([](AnyType type) {
    return conversion::utils::TVMFFIAnyLLVMDescriptor::getLLVMType(
        type.getContext());
  });
  typeConverter.addConversion([](ObjectHandleType type) {
    return conversion::utils::TVMFFIObjectHandleLLVMDescriptor::getLLVMType(
        type.getContext());
  });
  patterns.insert<LowerFunctionGetGlobalOp, LowerFunctionCallOp>(typeConverter,
                                                                 context);
  patterns.add<LowerGetTypeIndexOp, LowerToOp, LowerEnvTensorAllocOp,
               LowerObjectIncRefOp, LowerObjectDecRefOp,
               LowerErrorSetRaisedFromCStrOp, LowerGetOpaquePtrOp, LowerLoadOp,
               LowerStoreOp>(typeConverter, context);
  target.addIllegalDialect<TVMFFIDialect>();
}

void registerConvertTVMFFIToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx, TVMFFIDialect *dialect) {
    dialect->addInterfaces<TVMFFIToLLVMDialectInterface>();
  });
}

} // namespace libtriton::tvm_ffi
