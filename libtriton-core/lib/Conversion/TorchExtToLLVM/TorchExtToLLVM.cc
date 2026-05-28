#include <cstdint>

#include "dlpack/dlpack.h"
#include "libtriton-core/Conversion/TorchExtToLLVM/TorchExtToLLVM.h"
#include "libtriton-core/Conversion/Utils/RuntimeCFunctionDeclUtils.h"
#include "libtriton-core/Dialect/DLPack/IR/DLPackTypes.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

namespace libtriton::torch_ext {

#define GEN_PASS_DEF_CONVERTTORCHEXTTOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

namespace {

static bool isMemRefDescriptorLikeStructType(mlir::Type type) {
  auto structType = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(type);
  if (!structType) {
    return false;
  }
  llvm::ArrayRef<mlir::Type> body = structType.getBody();
  return body.size() == 5 && mlir::isa<mlir::LLVM::LLVMPointerType>(body[0]) &&
         mlir::isa<mlir::LLVM::LLVMPointerType>(body[1]) &&
         body[2].isInteger(64);
}

static mlir::Value
rewriteKernelOperand(mlir::ConversionPatternRewriter &rewriter,
                     mlir::Location loc, mlir::Value operand) {
  if (isMemRefDescriptorLikeStructType(operand.getType())) {
    return mlir::LLVM::ExtractValueOp::create(rewriter, loc, operand,
                                              llvm::ArrayRef<int64_t>{1})
        .getResult();
  } else {
    return operand;
  }
}

class ConvertKernelLaunchPattern
    : public mlir::OpConversionPattern<TritonKernelLaunchOp> {
public:
  using mlir::OpConversionPattern<TritonKernelLaunchOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(TritonKernelLaunchOp launchOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Location loc = launchOp.getLoc();

    llvm::SmallVector<mlir::Value> kernelOperands;
    kernelOperands.reserve(adaptor.getKernelOperands().size() + 2);
    for (mlir::Value operand : adaptor.getKernelOperands()) {
      kernelOperands.push_back(rewriteKernelOperand(rewriter, loc, operand));
    }

    mlir::LLVM::LLVMPointerType ptrType =
        mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    mlir::Value nullPtr = mlir::LLVM::ZeroOp::create(rewriter, loc, ptrType);
    kernelOperands.push_back(nullPtr);
    kernelOperands.push_back(nullPtr);

    // Use original (unconverted) grid/block size values so that gpu.launch_func
    // receives values of the expected type (index/i32/i64).
    mlir::gpu::KernelDim3 gridSize{launchOp.getGridSizeX(),
                                   launchOp.getGridSizeY(),
                                   launchOp.getGridSizeZ()};
    mlir::gpu::KernelDim3 blockSize{launchOp.getBlockSizeX(),
                                    launchOp.getBlockSizeY(),
                                    launchOp.getBlockSizeZ()};
    std::optional<mlir::gpu::KernelDim3> clusterSize = std::nullopt;
    if (launchOp.hasClusterSize()) {
      clusterSize = mlir::gpu::KernelDim3{launchOp.getClusterSizeX(),
                                          launchOp.getClusterSizeY(),
                                          launchOp.getClusterSizeZ()};
    }

    mlir::Type asyncTokenType;
    if (mlir::Value asyncToken = launchOp.getAsyncToken()) {
      asyncTokenType = asyncToken.getType();
    }
    llvm::SmallVector<mlir::Value> asyncDependencies =
        llvm::to_vector(adaptor.getAsyncDependencies());
    rewriter.replaceOpWithNewOp<mlir::gpu::LaunchFuncOp>(
        launchOp, adaptor.getKernelAttr(), gridSize, blockSize,
        adaptor.getDynamicSharedMemorySize(), kernelOperands, asyncTokenType,
        asyncDependencies, clusterSize);
    return mlir::success();
  }
};

class ConvertGetCurrentDevicePattern
    : public mlir::OpConversionPattern<GetCurrentDeviceOp> {
public:
  using mlir::OpConversionPattern<GetCurrentDeviceOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(GetCurrentDeviceOp op, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return mlir::failure();
    }

    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> calleeOrErr =
        libtriton::conversion::utils::runtime::getOrCreateGetCurrentDevice(
            moduleOp);
    if (mlir::failed(calleeOrErr)) {
      return mlir::failure();
    }

    mlir::Location loc = op.getLoc();
    mlir::LLVM::CallOp callOp = mlir::LLVM::CallOp::create(
        rewriter, loc, *calleeOrErr, mlir::ValueRange{});
    rewriter.replaceOp(op, callOp.getResults());
    return mlir::success();
  }
};

class ConvertGetCurrentStreamPattern
    : public mlir::OpConversionPattern<GetCurrentStreamOp> {
public:
  using mlir::OpConversionPattern<GetCurrentStreamOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(GetCurrentStreamOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return mlir::failure();
    }

    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> calleeOrErr =
        libtriton::conversion::utils::runtime::getOrCreateGetCurrentStream(
            moduleOp);
    if (mlir::failed(calleeOrErr)) {
      return mlir::failure();
    }

    mlir::Location loc = op.getLoc();
    mlir::Type i8Ty = mlir::IntegerType::get(rewriter.getContext(), 8);
    mlir::Value device = mlir::LLVM::ConstantOp::create(
        rewriter, loc, i8Ty, op.getDeviceAttr().getInt());
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, *calleeOrErr,
                                                    mlir::ValueRange{device});
    return mlir::success();
  }
};

class ConvertTorchExtToLLVMPass
    : public impl::ConvertTorchExtToLLVMBase<ConvertTorchExtToLLVMPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::LLVMTypeConverter typeConverter(&context);
    mlir::ConversionTarget target(context);
    mlir::RewritePatternSet patterns(&context);
    populateTorchExtToLLVMConversionPatterns(target, typeConverter, patterns);

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
    populateTorchExtToLLVMConversionPatterns(target, typeConverter, patterns);
  }
};

} // namespace

void populateTorchExtToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  typeConverter.addConversion(
      [](libtriton::dlpack::DLDeviceType type) -> mlir::Type {
        return mlir::LLVM::LLVMStructType::getLiteral(
            type.getContext(),
            {mlir::IntegerType::get(type.getContext(), 32),
             mlir::IntegerType::get(type.getContext(), 32)},
            /*isPacked=*/true);
      });
  patterns.add<ConvertGetCurrentDevicePattern, ConvertGetCurrentStreamPattern,
               ConvertKernelLaunchPattern>(typeConverter,
                                           patterns.getContext());
  target.addIllegalOp<GetCurrentDeviceOp>();
  target.addIllegalOp<GetCurrentStreamOp>();
  target.addIllegalOp<TritonKernelLaunchOp>();
  target.addLegalDialect<mlir::LLVM::LLVMDialect, mlir::gpu::GPUDialect>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
}

void registerConvertTorchExtToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx, TorchExtDialect *dialect) {
    dialect->addInterfaces<TorchExtToLLVMDialectInterface>();
  });
}

} // namespace libtriton::torch_ext
