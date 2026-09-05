//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/TorchExtToGPU/TorchExtToGPU.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "trident/core/Conversion/Utils/AOTICAPIDescriptors.h"
#include "trident/core/Conversion/Utils/TVMFFICAPIDescriptors.h"
#include "trident/core/Dialect/DLPack/IR/DLPackDialect.h"
#include "trident/core/Dialect/DLPack/IR/DLPackOps.h"
#include "trident/core/Dialect/DLPack/IR/DLPackTypes.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtAttrs.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TorchExt/IR/TorchExtOps.h"
#include <cstdint>
#include <dlpack/dlpack.h>
#include <llvm/ADT/APInt.h>
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
#include <torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h>
#include <torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h>
#include <tuple>
#include <utility>

namespace trident::conversion {

#define GEN_PASS_DEF_CONVERTTORCHEXTTOGPU
#include "trident/core/Conversion/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kSpecializationName = "triton.specialization";

} // namespace

/// Converts torch_ext.trident_kernel_launch to gpu.launch_func.
class ConvertTritonKernelLaunchOp final
    : public mlir::OpConversionPattern<torchext::TritonKernelLaunchOp> {
public:
  ConvertTritonKernelLaunchOp(mlir::TypeConverter &typeConverter,
                              mlir::MLIRContext *context)
      : mlir::OpConversionPattern<torchext::TritonKernelLaunchOp>(typeConverter,
                                                                  context) {}

  mlir::LogicalResult
  matchAndRewrite(torchext::TritonKernelLaunchOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::ArrayAttr const argAttrs = op.getArgAttrsAttr();
    llvm::SmallVector operandsAndSpecializations = llvm::map_to_vector(
        llvm::zip(argAttrs.getAsRange<mlir::DictionaryAttr>(),
                  adaptor.getKernelOperands()),
        [&](auto attrsAndOperand)
            -> std::tuple<mlir::Value, torchext::SpecializationAttr> {
          auto [argAttr, operand] = attrsAndOperand;
          torchext::SpecializationAttr const specialization =
              mlir::cast<torchext::SpecializationAttr>(
                  argAttr.get(kSpecializationName));
          mlir::Type const kind = specialization.getKind().getValue();
          if (operand.getType() != kind) {
            operand = getTypeConverter()->materializeTargetConversion(
                rewriter, loc, kind, operand);
          }
          return {operand, specialization};
        });
    if (llvm::any_of(operandsAndSpecializations, [](auto it) -> bool {
          auto [operand, specialization] = it;
          return !operand;
        })) {
      return op.emitOpError("failed to materialize a native kernel operand");
    }

    tvm_ffi::FuncOp function = op->getParentOfType<tvm_ffi::FuncOp>();
    if (function) {
      mlir::Value const allChecks = llvm::accumulate(
          operandsAndSpecializations,
          mlir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI1Type(), 1)
              .getResult(),
          [&](mlir::Value accumulated,
              const std::tuple<mlir::Value, torchext::SpecializationAttr> &it)
              -> mlir::Value {
            auto [operand, specialization] = it;
            uint64_t const divisibility = specialization.getDivisibility();
            if (divisibility == 1) {
              return accumulated;
            }
            mlir::Value const value =
                llvm::TypeSwitch<mlir::Type, mlir::Value>(operand.getType())
                    .Case<mlir::LLVM::LLVMPointerType>(
                        [&, operand = operand](
                            mlir::LLVM::LLVMPointerType) -> mlir::Value {
                          return mlir::LLVM::PtrToIntOp::create(
                              rewriter, loc, rewriter.getI64Type(), operand);
                        })
                    .Case<mlir::IntegerType>(
                        [&, operand = operand](
                            mlir::IntegerType type) -> mlir::Value {
                          return type.isInteger(64)
                                     ? operand
                                     : mlir::LLVM::SExtOp::create(
                                           rewriter, loc, rewriter.getI64Type(),
                                           operand)
                                           .getResult();
                        })
                    .Default([&, operand = operand](mlir::Type) -> mlir::Value {
                      return mlir::UnrealizedConversionCastOp::create(
                                 rewriter, loc, rewriter.getI64Type(), operand)
                          .getResult(0);
                    });

            mlir::Value const divisor = mlir::LLVM::ConstantOp::create(
                rewriter, loc, rewriter.getI64Type(),
                llvm::APInt(64, divisibility));
            mlir::Value const remainder =
                mlir::LLVM::URemOp::create(rewriter, loc, value, divisor);
            mlir::Value const zero = mlir::LLVM::ConstantOp::create(
                rewriter, loc, rewriter.getI64Type(), 0);
            // ICmpPredicate is generated into LLVMDialect.h without a
            // standalone public header.
            mlir::Value const check = mlir::LLVM::ICmpOp::create(
                rewriter, loc,
                mlir::LLVM::ICmpPredicate::eq, // NOLINT(misc-include-cleaner)
                remainder, zero);
            return mlir::arith::AndIOp::create(rewriter, loc, accumulated,
                                               check);
          });

      mlir::FunctionType const functionType = function.getFunctionType();
      if (functionType.getNumResults() != 1) {
        return op.emitOpError(
            "enclosing function must have one union result to report "
            "specialization failure");
      }
      auto resultUnion =
          mlir::dyn_cast<tvm_ffi::UnionType>(functionType.getResult(0));
      if (!resultUnion || !resultUnion.contains(tvm_ffi::ExceptionType::get(
                              function.getContext()))) {
        return op.emitOpError(
            "enclosing function result must contain !tvm_ffi.exception");
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
      mlir::Value const failureResult = tvm_ffi::CastOp::create(
          rewriter, loc, functionType.getResult(0), exception);
      tvm_ffi::ReturnOp::create(rewriter, loc, failureResult);
      rewriter.setInsertionPoint(op);
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
          auto [operand, specialization] = it;
          return operand;
        });
    mlir::Value const nullPointer =
        mlir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(), 0);
    operands.append(2, nullPointer);

    rewriter.replaceOpWithNewOp<mlir::gpu::LaunchFuncOp>(
        op, op.getKernel(), gridSize, blockSize, dynamicSharedMemorySize,
        operands, asyncObject, clusterSize);

    return mlir::success();
  }
};

class ConvertTorchExtToGPUPass final
    : public impl::ConvertTorchExtToGPUBase<ConvertTorchExtToGPUPass> {
public:
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    mlir::TypeConverter typeConverter;
    typeConverter.addConversion(
        [](mlir::Type type) -> std::optional<mlir::Type> {
          return llvm::TypeSwitch<mlir::Type, std::optional<mlir::Type>>(type)
              .Case<mlir::torch::Torch::NonValueTensorType,
                    mlir::torch::Torch::ValueTensorType>(
                  [](mlir::Type type) -> std::optional<mlir::Type> {
                    return mlir::LLVM::LLVMPointerType::get(type.getContext());
                  })
              .Case<mlir::torch::Torch::BoolType, mlir::torch::Torch::FloatType,
                    mlir::torch::Torch::IntType, mlir::IntegerType,
                    mlir::FloatType>(
                  [](mlir::Type type) -> std::optional<mlir::Type> {
                    return type;
                  })
              .Default([](mlir::Type) -> std::optional<mlir::Type> {
                return std::nullopt;
              });
        });
    typeConverter.addTargetMaterialization([](mlir::OpBuilder &builder,
                                              mlir::Type resultType,
                                              mlir::ValueRange inputs,
                                              mlir::Location loc)
                                               -> mlir::Value {
      mlir::Value const input = inputs.front();
      mlir::Type const inputType = input.getType();
      return llvm::TypeSwitch<mlir::Type, mlir::Value>(inputType)
          .Case<mlir::torch::Torch::BaseTensorType>(
              [&](mlir::Type) -> mlir::Value {
                if (!mlir::isa<mlir::LLVM::LLVMPointerType>(resultType)) {
                  return {};
                }
                mlir::Value const object = torchext::GetOp::create(
                    builder, loc,
                    tvm_ffi::ObjectType::get(builder.getContext()), input);
                mlir::Value const tensor = tvm_ffi::AsOp::create(
                    builder, loc,
                    dlpack::DLTensorType::get(builder.getContext()), object);
                return dlpack::TensorDataOp::create(builder, loc, resultType,
                                                    tensor)
                    .getResult();
              })
          .Case<mlir::torch::Torch::BoolType>([&](mlir::Type) -> mlir::Value {
            auto target = mlir::dyn_cast<mlir::IntegerType>(resultType);
            if (!target) {
              return {};
            }
            mlir::Value const value =
                mlir::torch::TorchConversion::ToI1Op::create(builder, loc,
                                                             input);
            return target.isInteger(1) ? value
                                       : mlir::LLVM::ZExtOp::create(
                                             builder, loc, resultType, value)
                                             .getResult();
          })
          .Case<mlir::torch::Torch::IntType>([&](mlir::Type) -> mlir::Value {
            auto target = mlir::dyn_cast<mlir::IntegerType>(resultType);
            if (!target) {
              return {};
            }
            mlir::Value const value =
                mlir::torch::TorchConversion::ToI64Op::create(builder, loc,
                                                              input);
            return target.isInteger(64) ? value
                                        : mlir::LLVM::TruncOp::create(
                                              builder, loc, resultType, value)
                                              .getResult();
          })
          .Case<mlir::torch::Torch::FloatType>([&](mlir::Type) -> mlir::Value {
            auto target = mlir::dyn_cast<mlir::FloatType>(resultType);
            if (!target) {
              return {};
            }
            mlir::Value const value =
                mlir::torch::TorchConversion::ToF64Op::create(builder, loc,
                                                              input);
            return target.isF64() ? value
                                  : mlir::LLVM::FPTruncOp::create(
                                        builder, loc, resultType, value)
                                        .getResult();
          })
          .Default([](mlir::Type) -> mlir::Value { return {}; });
    });

    target.addIllegalOp<torchext::TritonKernelLaunchOp>();
    target.addLegalOp<mlir::gpu::LaunchFuncOp, torchext::GetOp>();
    target.addLegalDialect<
        mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
        mlir::gpu::GPUDialect, mlir::BuiltinDialect, mlir::func::FuncDialect,
        mlir::LLVM::LLVMDialect, tvm_ffi::TVMFFIDialect, dlpack::DLPackDialect,
        mlir::torch::TorchConversion::TorchConversionDialect>();

    mlir::RewritePatternSet patterns(&getContext());
    populateTorchExtToGPUConversionPatterns(target, patterns, typeConverter);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

void populateTorchExtToGPUConversionPatterns(
    mlir::ConversionTarget &, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &typeConverter) {
  patterns.add<ConvertTritonKernelLaunchOp>(typeConverter,
                                            patterns.getContext());
}

} // namespace trident::conversion
