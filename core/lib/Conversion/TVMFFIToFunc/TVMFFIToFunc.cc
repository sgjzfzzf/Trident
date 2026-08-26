//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/TVMFFIToFunc/TVMFFIToFunc.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Conversion/Utils/GlobalString.h"
#include "trident/core/Conversion/Utils/TVMFFIUtils.h"
#include "trident/core/Conversion/Utils/Type.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtTypes.h"
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h> // NOLINT(misc-include-cleaner)
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <optional>
#include <string>
#include <torch-mlir/Dialect/Torch/IR/TorchTypes.h>

namespace trident::tvm_ffi {

#define GEN_PASS_DEF_CONVERTTVMFFITOFUNC
#include "trident/core/Conversion/Passes.h.inc"

namespace {

static std::optional<int32_t> getTVMFFITypeIndex(mlir::Type type) {
  if (mlir::isa<BoolType, mlir::torch::Torch::BoolType>(type)) {
    return BoolType::getTypeIndex();
  } else if (mlir::isa<IntType, mlir::torch::Torch::IntType>(type)) {
    return IntType::getTypeIndex();
  } else if (mlir::isa<FloatType, mlir::torch::Torch::FloatType>(type)) {
    return FloatType::getTypeIndex();
  } else if (mlir::isa<NoneType, mlir::torch::Torch::NoneType>(type)) {
    return NoneType::getTypeIndex();
  } else if (mlir::isa<StringType, mlir::torch::Torch::StringType>(type)) {
    return StringType::getTypeIndex();
  } else if (mlir::isa<DeviceType, mlir::torch::Torch::DeviceType>(type)) {
    return DeviceType::getTypeIndex();
  } else if (mlir::isa<DTypeType, trident::torchext::DTypeType>(type)) {
    return DTypeType::getTypeIndex();
  } else if (mlir::isa<ArrayType, mlir::torch::Torch::ListType,
                       mlir::torch::Torch::TupleType>(type)) {
    return ArrayType::getTypeIndex();
  } else if (mlir::isa<TensorType, mlir::torch::Torch::NonValueTensorType,
                       mlir::torch::Torch::ValueTensorType>(type)) {
    return TensorType::getTypeIndex();
  } else if (mlir::isa<FunctionType>(type)) {
    return FunctionType::getTypeIndex();
  } else {
    return std::nullopt;
  }
}

class ConvertTVMFFIToFuncPass final
    : public impl::ConvertTVMFFIToFuncBase<ConvertTVMFFIToFuncPass> {
  void runOnOperation() final {
    getOperation().walk([&](FuncOp tvmffiFuncOp) -> mlir::WalkResult {
      mlir::OpBuilder builder(tvmffiFuncOp);
      const mlir::FunctionType targetType = tvmffiFuncOp.getFunctionType();
      mlir::func::FuncOp funcOp =
          mlir::func::FuncOp::create(builder, tvmffiFuncOp.getLoc(),
                                     tvmffiFuncOp.getSymName(), targetType);
      if (tvmffiFuncOp.getSymVisibility()) {
        funcOp.setSymVisibilityAttr(tvmffiFuncOp.getSymVisibilityAttr());
      }
      mlir::IRMapping mapping;
      for (mlir::Block &sourceBlock : tvmffiFuncOp.getBody()) {
        const llvm::SmallVector<mlir::Location> argumentLocations =
            llvm::map_to_vector(
                sourceBlock.getArguments(),
                [](mlir::BlockArgument argument) { return argument.getLoc(); });
        mlir::Block *targetBlock = builder.createBlock(
            &funcOp.getBody(), funcOp.getBody().end(),
            sourceBlock.getArgumentTypes(), argumentLocations);
        mapping.map(&sourceBlock, targetBlock);
        mapping.map(sourceBlock.getArguments(), targetBlock->getArguments());
      }
      for (mlir::Block &sourceBlock : tvmffiFuncOp.getBody()) {
        mlir::Block *targetBlock = mapping.lookup(&sourceBlock);
        builder.setInsertionPointToEnd(targetBlock);
        for (mlir::Operation &sourceOperation : sourceBlock) {
          if (ReturnOp returnOp = mlir::dyn_cast<ReturnOp>(&sourceOperation)) {
            if (returnOp.getNumOperands() != targetType.getNumResults()) {
              returnOp.emitError("return value count does not match the "
                                 "converted function signature");
              signalPassFailure();
              return mlir::WalkResult::interrupt();
            }
            const llvm::SmallVector<mlir::Value> values = llvm::map_to_vector(
                returnOp.getOperands(), [&](mlir::Value value) {
                  return mapping.lookupOrDefault(value);
                });
            mlir::func::ReturnOp::create(builder, returnOp.getLoc(), values);
          } else {
            builder.clone(sourceOperation, mapping);
          }
        }
      }
      if (tvmffiFuncOp.getEmitTvmFfiAbi()) {
        mlir::MLIRContext *context = builder.getContext();
        mlir::ModuleOp module = tvmffiFuncOp->getParentOfType<mlir::ModuleOp>();
        if (!module) {
          tvmffiFuncOp.emitError("expected TVM FFI function inside a module");
          signalPassFailure();
          return mlir::WalkResult::interrupt();
        }
        const mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);
        const mlir::FunctionType wrapperType = mlir::FunctionType::get(
            context, {ptrTy, ptrTy, builder.getI32Type(), ptrTy},
            {builder.getI32Type()});
        const std::string wrapperName =
            llvm::formatv("__tvm_ffi_{0}", tvmffiFuncOp.getSymName()).str();
        builder.setInsertionPointAfter(funcOp);
        mlir::func::FuncOp wrapper = mlir::func::FuncOp::create(
            builder, tvmffiFuncOp.getLoc(), wrapperName, wrapperType);
        mlir::Block *entry = wrapper.addEntryBlock();
        const mlir::Value argsPtr = entry->getArgument(1);
        const mlir::Value resultPtr = entry->getArgument(3);
        const mlir::IntegerType i32Ty = builder.getI32Type();
        const mlir::IntegerType i64Ty = builder.getI64Type();
        mlir::LLVM::LLVMStructType llvmAnyTy =
            conversion::utils::getTVMFFIAnyType(context);
        mlir::LLVM::LLVMPointerType llvmPtrTy =
            mlir::LLVM::LLVMPointerType::get(context);

        builder.setInsertionPointToStart(entry);
        llvm::SmallVector<mlir::Value> arguments;
        mlir::Value allTypesMatch = mlir::arith::ConstantOp::create(
            builder, tvmffiFuncOp.getLoc(), builder.getI1Type(),
            builder.getBoolAttr(true));
        for (auto [index, type] :
             llvm::enumerate(tvmffiFuncOp.getArgumentTypes())) {
          std::optional<int32_t> expected = getTVMFFITypeIndex(type);
          if (!expected) {
            signalPassFailure();
            return mlir::WalkResult::interrupt();
          }
          const mlir::Value slot = mlir::LLVM::GEPOp::create(
              builder, tvmffiFuncOp.getLoc(), llvmPtrTy, llvmAnyTy, argsPtr,
              llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(index)});
          const mlir::Value value = mlir::LLVM::LoadOp::create(
              builder, tvmffiFuncOp.getLoc(), llvmAnyTy, slot);
          arguments.push_back(mlir::UnrealizedConversionCastOp::create(
                                  builder, tvmffiFuncOp.getLoc(), type, value)
                                  .getResult(0));
          const mlir::Value actual = mlir::LLVM::ExtractValueOp::create(
              builder, tvmffiFuncOp.getLoc(), i32Ty, value,
              llvm::ArrayRef<int64_t>{0});
          mlir::Value condition;
          if (mlir::isa<StringType, mlir::torch::Torch::StringType>(type)) {
            condition = mlir::arith::ConstantOp::create(
                builder, tvmffiFuncOp.getLoc(), builder.getI1Type(),
                builder.getBoolAttr(false));
            for (int32_t stringTypeIndex : {8, 11, 65}) {
              const mlir::Value isStringType = mlir::LLVM::ICmpOp::create(
                  builder, tvmffiFuncOp.getLoc(), mlir::LLVM::ICmpPredicate::eq,
                  actual,
                  mlir::LLVM::ConstantOp::create(builder, tvmffiFuncOp.getLoc(),
                                                 i32Ty, stringTypeIndex));
              condition = mlir::arith::OrIOp::create(
                  builder, tvmffiFuncOp.getLoc(), condition, isStringType);
            }
          } else {
            condition = mlir::LLVM::ICmpOp::create(
                builder, tvmffiFuncOp.getLoc(), mlir::LLVM::ICmpPredicate::eq,
                actual,
                mlir::LLVM::ConstantOp::create(builder, tvmffiFuncOp.getLoc(),
                                               i32Ty, expected.value()));
          }
          allTypesMatch = mlir::arith::AndIOp::create(
              builder, tvmffiFuncOp.getLoc(), allTypesMatch, condition);
        }
        mlir::scf::IfOp guard = mlir::scf::IfOp::create(
            builder, tvmffiFuncOp.getLoc(), allTypesMatch,
            /*withElseRegion=*/true);
        builder.setInsertionPointToStart(guard.thenBlock());
        mlir::func::CallOp call = mlir::func::CallOp::create(
            builder, tvmffiFuncOp.getLoc(), funcOp.getSymName(),
            funcOp.getFunctionType().getResults(), arguments);
        llvm::SmallVector<mlir::Value> results = call.getResults();
        if (results.size() == 1) {
          const mlir::Value value =
              mlir::UnrealizedConversionCastOp::create(
                  builder, tvmffiFuncOp.getLoc(), llvmAnyTy, results[0])
                  .getResult(0);
          mlir::LLVM::StoreOp::create(builder, tvmffiFuncOp.getLoc(), value,
                                      resultPtr);
        } else if (results.size() > 1) {
          mlir::Value slots = mlir::LLVM::AllocaOp::create(
              builder, tvmffiFuncOp.getLoc(), llvmPtrTy, llvmAnyTy,
              mlir::LLVM::ConstantOp::create(
                  builder, tvmffiFuncOp.getLoc(), i64Ty,
                  static_cast<int64_t>(results.size())));
          llvm::SmallVector<mlir::Value> const slotPtrs = llvm::map_to_vector(
              llvm::enumerate(results), [&](auto indexedResult) {
                auto [index, result] = indexedResult;
                mlir::Value const value =
                    mlir::UnrealizedConversionCastOp::create(
                        builder, tvmffiFuncOp.getLoc(), llvmAnyTy, result)
                        .getResult(0);
                const mlir::Value slot = mlir::LLVM::GEPOp::create(
                    builder, tvmffiFuncOp.getLoc(), llvmPtrTy, llvmAnyTy, slots,
                    llvm::ArrayRef<mlir::LLVM::GEPArg>{
                        static_cast<int32_t>(index)});
                mlir::LLVM::StoreOp::create(builder, tvmffiFuncOp.getLoc(),
                                            value, slot);
                return slot;
              });
          mlir::FailureOr<mlir::Value> array =
              conversion::utils::callTVMFFIGlobalFunction(
                  builder, tvmffiFuncOp.getLoc(), module, "ffi.Array",
                  slotPtrs);
          if (mlir::failed(array)) {
            tvmffiFuncOp.emitError("failed to create ffi.Array result");
            signalPassFailure();
            return mlir::WalkResult::interrupt();
          }
          const mlir::Value arrayValue = mlir::LLVM::LoadOp::create(
              builder, tvmffiFuncOp.getLoc(), llvmAnyTy, array.value());
          mlir::LLVM::StoreOp::create(builder, tvmffiFuncOp.getLoc(),
                                      arrayValue, resultPtr);
        }
        builder.setInsertionPointToStart(guard.elseBlock());
        const mlir::Value kindPtr = conversion::utils::getOrCreateGlobalString(
            builder, tvmffiFuncOp.getLoc(), module, "ExceptionKind",
            "GuardMatch");
        mlir::Value exceptionValue = mlir::LLVM::UndefOp::create(
            builder, tvmffiFuncOp.getLoc(), llvmAnyTy);
        exceptionValue = mlir::LLVM::InsertValueOp::create(
            builder, tvmffiFuncOp.getLoc(), exceptionValue,
            mlir::LLVM::ConstantOp::create(builder, tvmffiFuncOp.getLoc(),
                                           i32Ty, 8),
            llvm::ArrayRef<int64_t>{0});
        exceptionValue = mlir::LLVM::InsertValueOp::create(
            builder, tvmffiFuncOp.getLoc(), exceptionValue,
            mlir::LLVM::ConstantOp::create(builder, tvmffiFuncOp.getLoc(),
                                           i32Ty, 0),
            llvm::ArrayRef<int64_t>{1});
        exceptionValue = mlir::LLVM::InsertValueOp::create(
            builder, tvmffiFuncOp.getLoc(), exceptionValue,
            mlir::LLVM::PtrToIntOp::create(builder, tvmffiFuncOp.getLoc(),
                                           i64Ty, kindPtr),
            llvm::ArrayRef<int64_t>{2});
        const mlir::Value exceptionSlot = mlir::LLVM::AllocaOp::create(
            builder, tvmffiFuncOp.getLoc(), ptrTy, llvmAnyTy,
            mlir::LLVM::ConstantOp::create(builder, tvmffiFuncOp.getLoc(),
                                           i64Ty, 1));
        mlir::LLVM::StoreOp::create(builder, tvmffiFuncOp.getLoc(),
                                    exceptionValue, exceptionSlot);
        mlir::FailureOr<mlir::Value> exceptionResult =
            conversion::utils::callTVMFFIGlobalFunction(
                builder, tvmffiFuncOp.getLoc(), module, "trident.ffi.Exception",
                llvm::ArrayRef<mlir::Value>{exceptionSlot});
        if (mlir::failed(exceptionResult)) {
          tvmffiFuncOp.emitError("failed to create guard exception");
          signalPassFailure();
          return mlir::WalkResult::interrupt();
        }
        const mlir::Value exception =
            mlir::LLVM::LoadOp::create(builder, tvmffiFuncOp.getLoc(),
                                       llvmAnyTy, exceptionResult.value())
                .getResult();
        mlir::LLVM::StoreOp::create(builder, tvmffiFuncOp.getLoc(), exception,
                                    resultPtr);
        builder.setInsertionPointAfter(guard);
        const mlir::Value zero = mlir::LLVM::ConstantOp::create(
            builder, tvmffiFuncOp.getLoc(), i32Ty, 0);
        mlir::func::ReturnOp::create(builder, tvmffiFuncOp.getLoc(), zero);
      }
      tvmffiFuncOp->erase();
      return mlir::WalkResult::advance();
    });
  }
};

} // namespace
} // namespace trident::tvm_ffi
