//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/TVMFFIToGPU/TVMFFIToGPU.h"
#include "trident/core/Conversion/Utils/AOTICAPIDescriptors.h"
#include "trident/core/Conversion/Utils/TVMFFICAPIDescriptors.h"
#include "trident/core/Dialect/DLPack/IR/DLPackDialect.h"
#include "trident/core/Dialect/DLPack/IR/DLPackOps.h"
#include "trident/core/Dialect/DLPack/IR/DLPackTypes.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIAttrs.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIInterfaces.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <dlpack/dlpack.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <optional>
#include <tuple>
#include <utility>

namespace trident::conversion {

#define GEN_PASS_DEF_CONVERTTVMFFITOGPU
#include "trident/core/Conversion/Passes.h.inc"

/// Converts torch_ext.kernel_launch to gpu.launch_func.
class ConvertKernelLaunchOp final
    : public mlir::OpConversionPattern<tvm_ffi::KernelLaunchOp> {
public:
  ConvertKernelLaunchOp(mlir::TypeConverter &typeConverter,
                        mlir::MLIRContext *context)
      : mlir::OpConversionPattern<tvm_ffi::KernelLaunchOp>(typeConverter,
                                                           context) {}

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::KernelLaunchOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::ArrayAttr const specializations = op.getSpecializationsAttr();
    using OperandAndSpecialization =
        std::tuple<mlir::Value, tvm_ffi::SpecializationAttrInterface>;
    auto materializeOperand =
        [&](tvm_ffi::SpecializationAttrInterface specialization,
            mlir::Value source) -> OperandAndSpecialization {
      mlir::Value operand = source;
      mlir::Type const kind = specialization.getTargetType();
      if (operand.getType() != kind) {
        operand = getTypeConverter()->materializeTargetConversion(
            rewriter, loc, kind, operand);
      }
      return {operand, specialization};
    };

    tvm_ffi::FuncOp function = op->getParentOfType<tvm_ffi::FuncOp>();
    if (function) {
      llvm::SmallVector<OperandAndSpecialization> operandsAndSpecializations;
      for (auto [specialization, source] :
           llvm::zip(specializations
                         .getAsRange<tvm_ffi::SpecializationAttrInterface>(),
                     op.getKernelOperands())) {
        auto [operand, checkedSpecialization] =
            materializeOperand(specialization, source);
        if (!operand) {
          return op.emitOpError(
              "failed to materialize a native kernel operand");
        }
        operandsAndSpecializations.emplace_back(operand, checkedSpecialization);
      }
      mlir::Value const allChecks = llvm::accumulate(
          operandsAndSpecializations,
          mlir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 1)
              .getResult(),
          [&](mlir::Value accumulated,
              const OperandAndSpecialization &it) -> mlir::Value {
            auto [operand, specialization] = it;
            mlir::Value const check =
                specialization.buildCheck(rewriter, loc, operand);
            if (!check) {
              return accumulated;
            }
            return mlir::arith::AndIOp::create(rewriter, loc, accumulated,
                                               check);
          });

      mlir::FunctionType const functionType = function.getFunctionType();
      if (functionType.getNumResults() != 1) {
        return op.emitOpError(
            "enclosing function must have one result to report "
            "specialization failure");
      }
      mlir::Type const resultType = functionType.getResult(0);
      bool const canReportFailure =
          llvm::TypeSwitch<mlir::Type, bool>(resultType)
              .Case<tvm_ffi::AnyType>(
                  [](tvm_ffi::AnyType) -> bool { return true; })
              .Case<tvm_ffi::UnionType>(
                  [&](tvm_ffi::UnionType resultUnion) -> bool {
                    return resultUnion.contains(
                        tvm_ffi::ExceptionType::get(function.getContext()));
                  })
              .Default([](mlir::Type) -> bool { return false; });
      if (!canReportFailure) {
        return op.emitOpError(
            "enclosing function result must be !tvm_ffi.any or contain "
            "!tvm_ffi.exception");
      }

      mlir::Block *launchBlock = op->getBlock();
      mlir::Block *continuation =
          rewriter.splitBlock(launchBlock, op->getIterator());
      mlir::Block *failure =
          rewriter.createBlock(&function.getBody(), function.getBody().end());
      rewriter.setInsertionPointToEnd(launchBlock);
      mlir::cf::CondBranchOp::create(rewriter, loc, allChecks, continuation, {},
                                     failure, {});

      rewriter.setInsertionPointToEnd(failure);
      mlir::Value const exception = tvm_ffi::ExceptionOp::create(
          rewriter, loc, tvm_ffi::ExceptionType::get(function.getContext()),
          "GuardMatch");
      mlir::Value const failureResult =
          tvm_ffi::CastOp::create(rewriter, loc, resultType, exception);
      tvm_ffi::ReturnOp::create(rewriter, loc, failureResult);
      rewriter.setInsertionPoint(op);
    }
    // TODO: Materialized native values can borrow storage from their source
    // objects and must not outlive or cross blocks independently of them.
    // Add a block-transfer check for borrowed values so future conversions
    // cannot accidentally reuse the guard materialization in this block.
    llvm::SmallVector<OperandAndSpecialization> operandsAndSpecializations;
    for (auto [specialization, source] : llvm::make_filter_range(
             llvm::zip(specializations
                           .getAsRange<tvm_ffi::SpecializationAttrInterface>(),
                       op.getKernelOperands()),
             [](auto argument) -> bool {
               auto [specialization, _] = argument;
               return !mlir::isa<tvm_ffi::ConstantSpecializationAttr>(
                   specialization);
             })) {
      auto [operand, checkedSpecialization] =
          materializeOperand(specialization, source);
      if (!operand) {
        return op.emitOpError("failed to materialize a native kernel operand");
      }
      operandsAndSpecializations.emplace_back(operand, checkedSpecialization);
    }
    mlir::gpu::KernelDim3 const gridSize{
        adaptor.getGridSizeX(), adaptor.getGridSizeY(), adaptor.getGridSizeZ()};
    mlir::gpu::KernelDim3 const blockSize{adaptor.getBlockSizeX(),
                                          adaptor.getBlockSizeY(),
                                          adaptor.getBlockSizeZ()};

    std::optional<mlir::gpu::KernelDim3> clusterSize;
    if (adaptor.getClusterSizeX() && adaptor.getClusterSizeY() &&
        adaptor.getClusterSizeZ()) {
      clusterSize = mlir::gpu::KernelDim3{adaptor.getClusterSizeX(),
                                          adaptor.getClusterSizeY(),
                                          adaptor.getClusterSizeZ()};
    }

    mlir::Value const dynamicSharedMemorySize =
        adaptor.getDynamicSharedMemorySize();

    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return op->emitOpError("op is not inside a ModuleOp");
    }

    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getDeviceIndex =
        utils::getOrCreateAOTITorchGetCurrentDeviceIndex(moduleOp);
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getStream =
        utils::getOrCreateTVMFFIEnvGetStream(moduleOp);
    if (mlir::failed(getDeviceIndex) || mlir::failed(getStream)) {
      return op->emitOpError("failed to get the current CUDA stream");
    }

    mlir::IntegerType const i32 = rewriter.getI32Type();
    mlir::Value const deviceIndexSlot = mlir::LLVM::AllocaOp::create(
        rewriter, loc, mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
        i32,
        mlir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                       1));
    mlir::LLVM::CallOp::create(rewriter, loc, *getDeviceIndex, deviceIndexSlot);
    mlir::Value const deviceIndex =
        mlir::LLVM::LoadOp::create(rewriter, loc, i32, deviceIndexSlot);
    mlir::Value const cuda = mlir::LLVM::ConstantOp::create(
        rewriter, loc, i32, DLDeviceType::kDLCUDA);
    mlir::Value const asyncObject =
        mlir::LLVM::CallOp::create(rewriter, loc, *getStream,
                                   mlir::ValueRange{cuda, deviceIndex})
            .getResult();

    llvm::SmallVector<mlir::Value> operands = llvm::map_to_vector(
        operandsAndSpecializations, [](auto it) -> mlir::Value {
          auto [operand, _] = it;
          return operand;
        });
    mlir::Value const nullPointer =
        mlir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(), 0);
    operands.append(2, nullPointer);

    rewriter.replaceOpWithNewOp<mlir::gpu::LaunchFuncOp>(
        op, op.getKernel(), gridSize, blockSize, dynamicSharedMemorySize,
        operands, nullptr, mlir::ValueRange{}, asyncObject, clusterSize);

    return mlir::success();
  }
};

class ConvertTVMFFIToGPUPass final
    : public impl::ConvertTVMFFIToGPUBase<ConvertTVMFFIToGPUPass> {
public:
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    mlir::TypeConverter typeConverter;
    typeConverter.addConversion(
        [](mlir::Type type) -> mlir::Type { return type; });
    typeConverter.addTargetMaterialization([](mlir::OpBuilder &builder,
                                              mlir::Type resultType,
                                              mlir::ValueRange inputs,
                                              mlir::Location loc)
                                               -> mlir::Value {
      if (inputs.size() != 1) {
        return {};
      }
      mlir::Value input = inputs.front();
      if (mlir::isa<tvm_ffi::TensorType>(input.getType())) {
        if (!mlir::isa<mlir::LLVM::LLVMPointerType>(resultType)) {
          return {};
        }
        mlir::Value const object = tvm_ffi::GetOp::create(
            builder, loc, tvm_ffi::ObjectType::get(builder.getContext()),
            input);
        mlir::Value const tensor = tvm_ffi::AsOp::create(
            builder, loc, dlpack::DLTensorType::get(builder.getContext()),
            object);
        return dlpack::TensorDataOp::create(builder, loc, resultType, tensor);
      }
      if (mlir::isa<tvm_ffi::BoolType, tvm_ffi::IntType, tvm_ffi::FloatType>(
              input.getType())) {
        input = tvm_ffi::GetOp::create(builder, loc, input);
      }
      mlir::Type const inputType = input.getType();
      if (inputType == resultType) {
        return input;
      }
      if (auto integer = mlir::dyn_cast<mlir::IntegerType>(inputType)) {
        auto targetInteger = mlir::dyn_cast<mlir::IntegerType>(resultType);
        if (!targetInteger) {
          return {};
        }
        if (integer.getWidth() > targetInteger.getWidth()) {
          return mlir::arith::TruncIOp::create(builder, loc, resultType, input);
        }
        if (integer.isInteger(1)) {
          return mlir::arith::ExtUIOp::create(builder, loc, resultType, input);
        }
        return mlir::arith::ExtSIOp::create(builder, loc, resultType, input);
      }
      if (mlir::isa<mlir::FloatType>(inputType) &&
          mlir::isa<mlir::FloatType>(resultType)) {
        return mlir::arith::TruncFOp::create(builder, loc, resultType, input);
      }
      return {};
    });

    target.addIllegalOp<tvm_ffi::KernelLaunchOp>();
    target.addLegalDialect<mlir::arith::ArithDialect,
                           mlir::cf::ControlFlowDialect, mlir::gpu::GPUDialect,
                           mlir::BuiltinDialect, mlir::func::FuncDialect,
                           mlir::LLVM::LLVMDialect, tvm_ffi::TVMFFIDialect,
                           dlpack::DLPackDialect>();

    mlir::RewritePatternSet patterns(&getContext());
    populateTVMFFIToGPUConversionPatterns(target, patterns, typeConverter);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

void populateTVMFFIToGPUConversionPatterns(mlir::ConversionTarget &,
                                           mlir::RewritePatternSet &patterns,
                                           mlir::TypeConverter &typeConverter) {
  patterns.add<ConvertKernelLaunchOp>(typeConverter, patterns.getContext());
}

} // namespace trident::conversion
