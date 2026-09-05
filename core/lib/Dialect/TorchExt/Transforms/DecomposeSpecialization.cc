//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TorchExt/Transforms/DecomposeSpecialization.h"

#include <cstdint>
#include <mlir/Support/LLVM.h>
#include <optional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "trident/core/Dialect/DLPack/IR/DLPackDialect.h"
#include "trident/core/Dialect/DLPack/IR/DLPackOps.h"
#include "trident/core/Dialect/DLPack/IR/DLPackTypes.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtAttrs.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "llvm/ADT/STLExtras.h"

namespace trident::torchext {

#define GEN_PASS_DEF_DECOMPOSESPECIALIZATION
#include "trident/core/Dialect/TorchExt/Transforms/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kArgAttrsName = "arg_attrs";
constexpr llvm::StringLiteral kSpecializationName = "triton.specialization";

mlir::Value materializeTensorDataPointer(mlir::OpBuilder &builder,
                                         mlir::Location loc,
                                         mlir::Value object) {
  mlir::Value tensor = tvm_ffi::AsOp::create(
      builder, loc, dlpack::DLTensorType::get(builder.getContext()), object);
  return dlpack::TensorDataOp::create(
             builder, loc,
             mlir::LLVM::LLVMPointerType::get(builder.getContext()), tensor)
      .getResult();
}

mlir::FailureOr<mlir::Value> materializeNativeValue(mlir::OpBuilder &builder,
                                                    mlir::Location loc,
                                                    mlir::Value operand) {
  mlir::Type type = operand.getType();
  if (mlir::isa<mlir::torch::Torch::BaseTensorType>(type)) {
    mlir::Value object = GetOp::create(
        builder, loc, tvm_ffi::ObjectType::get(builder.getContext()), operand);
    return materializeTensorDataPointer(builder, loc, object);
  }
  if (mlir::isa<tvm_ffi::TensorType>(type)) {
    mlir::Value object = tvm_ffi::GetOp::create(
        builder, loc, tvm_ffi::ObjectType::get(builder.getContext()), operand);
    return materializeTensorDataPointer(builder, loc, object);
  }
  if (mlir::isa<mlir::torch::Torch::IntType>(type)) {
    return GetOp::create(builder, loc, builder.getI64Type(), operand)
        .getResult();
  }
  if (mlir::isa<tvm_ffi::IntType>(type)) {
    return tvm_ffi::GetOp::create(builder, loc, builder.getI64Type(), operand)
        .getResult();
  }
  if (mlir::isa<mlir::LLVM::LLVMPointerType, mlir::IntegerType>(type)) {
    return operand;
  }
  return mlir::failure();
}

mlir::FailureOr<mlir::Value>
convertToI64(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value) {
  mlir::IntegerType i64 = builder.getI64Type();
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType())) {
    return mlir::LLVM::PtrToIntOp::create(builder, loc, i64, value).getResult();
  }

  mlir::IntegerType integerType =
      mlir::dyn_cast<mlir::IntegerType>(value.getType());
  if (!integerType) {
    return mlir::failure();
  }

  uint32_t width = integerType.getWidth();
  if (width == 64) {
    return value;
  }
  if (width < 64) {
    return mlir::LLVM::SExtOp::create(builder, loc, i64, value).getResult();
  }
  return mlir::failure();
}

mlir::FailureOr<mlir::Value> buildDivisibilityCheck(mlir::OpBuilder &builder,
                                                    mlir::Location loc,
                                                    mlir::Value operand,
                                                    int64_t divisibility) {
  mlir::FailureOr<mlir::Value> nativeValue =
      materializeNativeValue(builder, loc, operand);
  if (mlir::failed(nativeValue)) {
    return mlir::failure();
  }
  mlir::FailureOr<mlir::Value> value = convertToI64(builder, loc, *nativeValue);
  if (mlir::failed(value)) {
    return mlir::failure();
  }

  mlir::IntegerType i64 = builder.getI64Type();
  mlir::Value divisor = mlir::LLVM::ConstantOp::create(
      builder, loc, i64, static_cast<int64_t>(divisibility));
  mlir::Value remainder =
      mlir::LLVM::URemOp::create(builder, loc, *value, divisor);
  mlir::Value zero = mlir::LLVM::ConstantOp::create(builder, loc, i64, 0);
  return mlir::LLVM::ICmpOp::create(builder, loc, mlir::LLVM::ICmpPredicate::eq,
                                    remainder, zero)
      .getResult();
}

mlir::LogicalResult createFailurePath(TridentKernelLaunchOp launch,
                                      mlir::Value condition) {
  mlir::func::FuncOp function = launch->getParentOfType<mlir::func::FuncOp>();
  if (!function) {
    return launch.emitOpError(
        "must be nested in a func.func to report specialization failure");
  }

  mlir::FunctionType functionType = function.getFunctionType();
  if (functionType.getNumResults() != 1) {
    return launch.emitOpError(
        "enclosing function must have one union result to report "
        "specialization failure");
  }
  auto resultUnion =
      mlir::dyn_cast<tvm_ffi::UnionType>(functionType.getResult(0));
  if (!resultUnion || !resultUnion.contains(
                          tvm_ffi::ExceptionType::get(function.getContext()))) {
    return launch.emitOpError(
        "enclosing function result must contain !tvm_ffi.exception");
  }

  mlir::Block *launchBlock = launch->getBlock();
  mlir::Block *continuation = launchBlock->splitBlock(launch);
  mlir::OpBuilder builder(function.getContext());
  mlir::Block *failure =
      builder.createBlock(&function.getBody(), function.getBody().end());

  builder.setInsertionPointToEnd(launchBlock);
  mlir::cf::CondBranchOp::create(builder, launch.getLoc(), condition,
                                 continuation, {}, failure, {});

  builder.setInsertionPointToEnd(failure);
  mlir::Value exception = tvm_ffi::ExceptionOp::create(
      builder, launch.getLoc(),
      tvm_ffi::ExceptionType::get(function.getContext()), "GuardMatch");
  mlir::Value failureResult = tvm_ffi::CastOp::create(
      builder, launch.getLoc(), functionType.getResult(0), exception);
  mlir::func::ReturnOp::create(builder, launch.getLoc(), failureResult);
  return mlir::success();
}

mlir::LogicalResult decomposeLaunch(TridentKernelLaunchOp launch) {
  mlir::ArrayAttr argAttrs = launch.getArgAttrsAttr();
  if (!argAttrs) {
    return mlir::failure();
  }
  if (argAttrs.size() != launch.getKernelOperands().size()) {
    return launch.emitOpError(
        "arg_attrs and kernel operands must have the same size");
  }

  mlir::OpBuilder builder(launch);
  std::optional<mlir::Value> allChecks;
  llvm::SmallVector<mlir::Attribute> updatedArgAttrs(argAttrs.begin(),
                                                     argAttrs.end());

  for (auto [operandIndex, attrsAndOperand] :
       llvm::enumerate(llvm::zip(argAttrs, launch.getKernelOperands()))) {
    auto [attribute, operand] = attrsAndOperand;
    mlir::DictionaryAttr argAttr = mlir::cast<mlir::DictionaryAttr>(attribute);
    auto specialization = mlir::dyn_cast_or_null<SpecializationAttr>(
        argAttr.get(kSpecializationName));
    if (!specialization) {
      continue;
    }

    int64_t divisibility = specialization.getDivisibility();
    if (divisibility <= 0) {
      return launch.emitOpError("invalid Triton specialization entry");
    }
    mlir::FailureOr<mlir::Value> check =
        buildDivisibilityCheck(builder, launch.getLoc(), operand, divisibility);
    if (mlir::failed(check)) {
      return launch.emitOpError(
          "unsupported operand type for Triton specialization");
    }
    allChecks = allChecks ? mlir::arith::AndIOp::create(
                                builder, launch.getLoc(), *allChecks, *check)
                          : *check;

    llvm::SmallVector<mlir::NamedAttribute> remainingAttrs =
        llvm::filter_to_vector(
            argAttr.getValue(), [](mlir::NamedAttribute namedAttr) -> bool {
              return namedAttr.getName() != kSpecializationName ||
                     !mlir::isa<SpecializationAttr>(namedAttr.getValue());
            });
    updatedArgAttrs[operandIndex] =
        mlir::DictionaryAttr::get(builder.getContext(), remainingAttrs);
  }

  if (!allChecks) {
    return mlir::success();
  }

  if (mlir::failed(createFailurePath(launch, *allChecks))) {
    return mlir::failure();
  }

  launch->setAttr(kArgAttrsName,
                  mlir::ArrayAttr::get(builder.getContext(), updatedArgAttrs));
  return mlir::success();
}

class DecomposeSpecializationPass final
    : public impl::DecomposeSpecializationBase<DecomposeSpecializationPass> {
  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    llvm::SmallVector<TridentKernelLaunchOp> launches;
    module.walk([&](TridentKernelLaunchOp launch) {
      if (launch->hasAttr(kArgAttrsName)) {
        launches.push_back(launch);
      }
    });

    for (TridentKernelLaunchOp launch : launches) {
      if (mlir::failed(decomposeLaunch(launch))) {
        signalPassFailure();
      }
    }
  }
};

} // namespace

} // namespace trident::torchext
