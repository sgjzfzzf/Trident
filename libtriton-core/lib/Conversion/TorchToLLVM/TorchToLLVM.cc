#include "libtriton-core/Conversion/TorchToLLVM/TorchToLLVM.h"
#include "SchemaLookup.h"
#include "libtriton-core/Conversion/Utils/TVMFFIUtils.h"
#include "libtriton-core/Conversion/Utils/Type.h"
#include "libtriton-core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Rewrite.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch/csrc/stable/c/shim.h"
#include "tvm/ffi/c_api.h"
#include "llvm/ADT/SmallVectorExtras.h"

namespace libtriton::torch {

#define GEN_PASS_DEF_CONVERTTORCHTOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Lower torch.constant.* ops directly to LLVM::ConstantOp
// ---------------------------------------------------------------------------

/// Lowers torch.constant.bool to LLVM TVMFFIAny {kTVMFFIBool, 0, value}.
class ConvertTorchConstantBoolOp
    : public mlir::OpConversionPattern<mlir::torch::Torch::ConstantBoolOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = mlir::torch::Torch::ConstantBoolOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ConstantBoolOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type anyTy = getTypeConverter()->convertType(op.getType());
    mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);

    mlir::Value result = mlir::LLVM::UndefOp::create(rewriter, loc, anyTy);
    mlir::Value typeIdx =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, kTVMFFIBool);
    result = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, anyTy, result, typeIdx, llvm::ArrayRef<int64_t>{0});
    mlir::Value payload =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, op.getValue());
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        op, anyTy, result, payload, llvm::ArrayRef<int64_t>{2});
    return mlir::success();
  }
};

/// Lowers torch.constant.int to LLVM TVMFFIAny {kTVMFFIInt, 0, value}.
class ConvertTorchConstantIntOp
    : public mlir::OpConversionPattern<mlir::torch::Torch::ConstantIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = mlir::torch::Torch::ConstantIntOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ConstantIntOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type anyTy = getTypeConverter()->convertType(op.getType());
    mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);

    mlir::Value result = mlir::LLVM::UndefOp::create(rewriter, loc, anyTy);
    mlir::Value typeIdx =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, kTVMFFIInt);
    result = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, anyTy, result, typeIdx, llvm::ArrayRef<int64_t>{0});
    mlir::Value payload =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, op.getValue());
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        op, anyTy, result, payload, llvm::ArrayRef<int64_t>{2});
    return mlir::success();
  }
};

/// Lowers torch.constant.float to LLVM TVMFFIAny {kTVMFFIFloat, 0, value}.
class ConvertTorchConstantFloatOp
    : public mlir::OpConversionPattern<mlir::torch::Torch::ConstantFloatOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = mlir::torch::Torch::ConstantFloatOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ConstantFloatOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type anyTy = getTypeConverter()->convertType(op.getType());
    mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);

    mlir::Value result = mlir::LLVM::UndefOp::create(rewriter, loc, anyTy);
    mlir::Value typeIdx =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, kTVMFFIFloat);
    result = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, anyTy, result, typeIdx, llvm::ArrayRef<int64_t>{0});
    mlir::Value floatVal = mlir::LLVM::ConstantOp::create(
        rewriter, loc, mlir::Float64Type::get(ctx), op.getValue());
    mlir::Value payload =
        mlir::LLVM::BitcastOp::create(rewriter, loc, i64Ty, floatVal);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        op, anyTy, result, payload, llvm::ArrayRef<int64_t>{2});
    return mlir::success();
  }
};

/// Lowers torch.constant.none to LLVM TVMFFIAny {kTVMFFINone, 0, 0}.
class ConvertTorchConstantNoneOp
    : public mlir::OpConversionPattern<mlir::torch::Torch::ConstantNoneOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = mlir::torch::Torch::ConstantNoneOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ConstantNoneOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type anyTy = getTypeConverter()->convertType(op.getType());
    mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);

    mlir::Value result = mlir::LLVM::UndefOp::create(rewriter, loc, anyTy);
    mlir::Value typeIdx =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, kTVMFFINone);
    result = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, anyTy, result, typeIdx, llvm::ArrayRef<int64_t>{0});
    mlir::Value payload =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 0);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        op, anyTy, result, payload, llvm::ArrayRef<int64_t>{2});
    return mlir::success();
  }
};

/// Lowers torch.constant.device to LLVM TVMFFIAny {kTVMFFIDevice, 0, combined}.
/// The device value is encoded as (device_index << 32) | device_type in the
/// i64 payload, matching the TVM FFI kTVMFFIDevice convention.
class ConvertTorchConstantDeviceOp
    : public mlir::OpConversionPattern<mlir::torch::Torch::ConstantDeviceOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = mlir::torch::Torch::ConstantDeviceOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ConstantDeviceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type anyTy = getTypeConverter()->convertType(op.getType());
    mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);

    // Parse the device string (e.g. "cuda:0", "cpu") using PyTorch's
    // stable C API.
    llvm::StringRef dev = op.getValue();
    uint32_t deviceType = 0;
    int32_t deviceIndex = 0;
    if (torch_parse_device_string(dev.data(), &deviceType, &deviceIndex)) {
      return op.emitError("failed to parse device string: ") << dev;
    }

    // Build TVMFFIAny: {kTVMFFIDevice=6, 0, (device_index<<32)|device_type}.
    mlir::Value result = mlir::LLVM::UndefOp::create(rewriter, loc, anyTy);
    mlir::Value typeIdx =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, kTVMFFIDevice);
    result = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, anyTy, result, typeIdx, llvm::ArrayRef<int64_t>{0});

    // Encode: (device_index << 32) | device_type.
    mlir::Value devType64 =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, deviceType);
    mlir::Value devIdx64 =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, deviceIndex);
    mlir::Value shifted = mlir::LLVM::ShlOp::create(
        rewriter, loc, devIdx64,
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 32));
    mlir::Value combined =
        mlir::LLVM::OrOp::create(rewriter, loc, devType64, shifted);

    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        op, anyTy, result, combined, llvm::ArrayRef<int64_t>{2});
    return mlir::success();
  }
};

/// Converts torch.aten.* ops directly to aoti_torch_call_dispatcher() LLVM
/// calls, preserving the original Torch dialect op schema for downstream
/// inspection. The op name is regex-matched to extract the dispatcher
/// op_name (e.g. "aten::empty_like") and optional overload_name.
class ConvertAtenDispatcherOp : public mlir::ConversionPattern {
public:
  ConvertAtenDispatcherOp(const mlir::TypeConverter &typeConverter,
                          mlir::MLIRContext *context)
      : mlir::ConversionPattern(typeConverter,
                                mlir::Pattern::MatchAnyOpTypeTag(),
                                /*benefit=*/1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Wrap adapted operands for the C API.
    llvm::SmallVector<MlirValue> wrappedOperands =
        llvm::map_to_vector(operands, [](mlir::Value v) { return wrap(v); });

    // Allocate output array (handles zero results gracefully).
    llvm::SmallVector<MlirValue> mlirResults(op->getNumResults(), {nullptr});

    // Delegate the full lowering to SchemaLookup.
    if (LibTritonSchemaDispatchTorchAtenOp(wrap(op), wrappedOperands.data(),
                                           mlirResults.data(),
                                           wrap(&rewriter))) {
      return mlir::failure();
    }

    // Unwrap results and hand them back to the dialect conversion framework.
    llvm::SmallVector<mlir::Value> results = llvm::map_to_vector(
        mlirResults, [](MlirValue mv) { return unwrap(mv); });
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

// ---------------------------------------------------------------------------
// StableList operations
// ---------------------------------------------------------------------------

/// Converts torch.prim.ListConstruct.
///
/// Constructs an ffi.Array via callTVMFFIGlobalFunction with all elements
/// passed as packed args.  The result is a TVMFFIObjectHandle (!llvm.ptr)
/// which is later converted to a StableListHandle in SchemaLookup when the
/// list reaches an Aten dispatcher call.
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
    mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);

    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return op.emitError("op is not inside a module");
    }

    // Adapted elements are now TVMFFIAny. Extract i64 payload from each.
    mlir::ValueRange elements = adaptor.getElements();
    mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
    mlir::LLVM::LLVMStructType anyTy =
        libtriton::conversion::utils::getTVMFFIAnyType(ctx);
    const size_t N = elements.size();
    mlir::Value ffiArgs = mlir::LLVM::AllocaOp::create(
        rewriter, loc, ptrTy, anyTy,
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, N));
    mlir::Value kTVMFFIIntVal =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, kTVMFFIInt);
    mlir::Value zero32 =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, 0);
    for (auto [i, element] : llvm::enumerate(elements)) {
      // Extract i64 payload from the TVMFFIAny element.
      mlir::Value elemPayload = mlir::LLVM::ExtractValueOp::create(
          rewriter, loc, element, llvm::ArrayRef<int64_t>{2});

      mlir::Value slot =
          mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, anyTy, ffiArgs,
                                    llvm::ArrayRef<mlir::LLVM::GEPArg>{i});
      mlir::LLVM::StoreOp::create(
          rewriter, loc, kTVMFFIIntVal,
          mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, anyTy, slot,
                                    llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0}));
      mlir::LLVM::StoreOp::create(
          rewriter, loc, zero32,
          mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, anyTy, slot,
                                    llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1}));
      mlir::LLVM::StoreOp::create(
          rewriter, loc, elemPayload,
          mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, anyTy, slot,
                                    llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2}));
    }

    // Call ffi.Array(elem0, ..., elemN) — pass each slot individually.
    llvm::SmallVector<mlir::Value> slotPtrs =
        llvm::map_to_vector(llvm::seq(N), [&](size_t i) -> mlir::Value {
          return mlir::LLVM::GEPOp::create(
              rewriter, loc, ptrTy, anyTy, ffiArgs,
              llvm::ArrayRef<mlir::LLVM::GEPArg>{i});
        });
    mlir::FailureOr<mlir::Value> result =
        libtriton::conversion::utils::callTVMFFIGlobalFunction(
            rewriter, loc, moduleOp, "ffi.Array", slotPtrs);
    if (mlir::failed(result)) {
      return mlir::failure();
    }

    // Extract v_obj (field[2]) from result TVMFFIAny and wrap it back
    // in a TVMFFIAny with kTVMFFIArray tag so downstream consumers
    // (SchemaLookup for aten ops, ConvertListDeleteListOp, etc.) always
    // see a proper TVMFFIAny value instead of a raw pointer that would
    // force an unreconcilable unrealized_conversion_cast.
    mlir::Value resultSlot = *result;
    mlir::Value vObjGEP =
        mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, anyTy, resultSlot,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    mlir::Value vObj =
        mlir::LLVM::LoadOp::create(rewriter, loc, i64Ty, vObjGEP);

    // Build TVMFFIAny {kTVMFFIArray, 0, vObj}.
    mlir::Value anyResult = mlir::LLVM::UndefOp::create(rewriter, loc, anyTy);
    mlir::Value kTVMFFIArrayVal =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, kTVMFFIArray);
    anyResult = mlir::LLVM::InsertValueOp::create(rewriter, loc, anyTy,
                                                  anyResult, kTVMFFIArrayVal,
                                                  llvm::ArrayRef<int64_t>{0});
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        op, anyTy, anyResult, vObj, llvm::ArrayRef<int64_t>{2});
    return mlir::success();
  }
};

// ---------------------------------------------------------------------------
// ConvertTorchToLLVM pass
// ---------------------------------------------------------------------------

class ConvertTorchToLLVMPass
    : public impl::ConvertTorchToLLVMBase<ConvertTorchToLLVMPass> {
public:
  void runOnOperation() final {
    mlir::LLVMTypeConverter typeConverter(&getContext());
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());

    populateTorchToLLVMConversionPatterns(target, typeConverter, patterns);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

// ---------------------------------------------------------------------------
// ConvertToLLVM interface for Torch dialect
// ---------------------------------------------------------------------------

struct TorchToLLVMDialectInterface
    : public mlir::ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(
      mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
      mlir::RewritePatternSet &patterns) const final {
    populateTorchToLLVMConversionPatterns(target, typeConverter, patterns);
  }
};

} // namespace

void populateTorchToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  setupBackendTypeConversion(target, typeConverter);
  patterns.add<ConvertAtenDispatcherOp, ConvertTorchConstantBoolOp,
               ConvertTorchConstantIntOp, ConvertTorchConstantFloatOp,
               ConvertTorchConstantNoneOp, ConvertTorchConstantDeviceOp,
               ConvertPrimListConstructOp>(typeConverter,
                                           patterns.getContext());
  target.addIllegalOp<
      mlir::torch::Torch::ConstantBoolOp, mlir::torch::Torch::ConstantIntOp,
      mlir::torch::Torch::ConstantFloatOp, mlir::torch::Torch::ConstantNoneOp,
      mlir::torch::Torch::ConstantDeviceOp,
      mlir::torch::Torch::PrimListConstructOp>();
  target.addLegalDialect<mlir::LLVM::LLVMDialect, mlir::BuiltinDialect,
                         mlir::func::FuncDialect>();
}

void registerConvertTorchToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::torch::Torch::TorchDialect *dialect) {
        dialect->addInterfaces<TorchToLLVMDialectInterface>();
      });
}

} // namespace libtriton::torch
