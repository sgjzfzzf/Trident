//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/TVMFFIToFunc/TVMFFIToFunc.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Conversion/Utils/String.h"
#include "trident/core/Conversion/Utils/TVMFFIUtils.h"
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
#include <string>
#include <torch-mlir/Dialect/Torch/IR/TorchTypes.h>

namespace trident::conversion {

#define GEN_PASS_DEF_CONVERTTVMFFITOFUNC
#include "trident/core/Conversion/Passes.h.inc"

namespace {

llvm::SmallVector<int32_t> getExpectedTypeIndices(mlir::Type type) {
  if (const tvm_ffi::UnionType unionType =
          mlir::dyn_cast<tvm_ffi::UnionType>(type)) {
    return llvm::map_to_vector(
        unionType.getTypes(), [](mlir::Type member) -> int32_t {
          return mlir::cast<tvm_ffi::TVMFFITypeIndexInterface>(member)
              .getTypeIndex();
        });
  }
  if (mlir::isa<tvm_ffi::AnyType>(type)) {
    return {};
  }
  if (mlir::isa<tvm_ffi::TVMFFIABIType>(type)) {
    return {mlir::cast<tvm_ffi::TVMFFITypeIndexInterface>(type).getTypeIndex()};
  }
  if (mlir::isa<mlir::torch::Torch::BoolType>(type)) {
    return {tvm_ffi::BoolType::getTypeIndex()};
  } else if (mlir::isa<mlir::torch::Torch::IntType>(type)) {
    return {tvm_ffi::IntType::getTypeIndex()};
  } else if (mlir::isa<mlir::torch::Torch::FloatType>(type)) {
    return {tvm_ffi::FloatType::getTypeIndex()};
  } else if (mlir::isa<mlir::torch::Torch::NoneType>(type)) {
    return {tvm_ffi::NoneType::getTypeIndex()};
  } else if (mlir::isa<mlir::torch::Torch::StringType>(type)) {
    return {tvm_ffi::RawStrType::getTypeIndex(),
            tvm_ffi::SmallStrType::getTypeIndex(),
            tvm_ffi::StrType::getTypeIndex()};
  } else if (mlir::isa<mlir::torch::Torch::DeviceType>(type)) {
    return {tvm_ffi::DeviceType::getTypeIndex()};
  } else if (mlir::isa<torchext::DTypeType>(type)) {
    return {tvm_ffi::DTypeType::getTypeIndex()};
  } else if (mlir::isa<mlir::torch::Torch::ListType,
                       mlir::torch::Torch::TupleType>(type)) {
    return {tvm_ffi::ArrayType::getTypeIndex()};
  } else if (mlir::isa<mlir::torch::Torch::NonValueTensorType,
                       mlir::torch::Torch::ValueTensorType>(type)) {
    return {tvm_ffi::TensorType::getTypeIndex()};
  }
  return {};
}

} // namespace

class ConvertTVMFFIToFuncPass final
    : public impl::ConvertTVMFFIToFuncBase<ConvertTVMFFIToFuncPass> {
  void runOnOperation() final {
    getOperation().walk([&](tvm_ffi::FuncOp tvmffiFuncOp) -> mlir::WalkResult {
      mlir::OpBuilder builder(tvmffiFuncOp);
      const mlir::FunctionType targetType = tvmffiFuncOp.getFunctionType();
      mlir::func::FuncOp funcOp =
          mlir::func::FuncOp::create(builder, tvmffiFuncOp.getLoc(),
                                     tvmffiFuncOp.getSymName(), targetType);
      if (tvmffiFuncOp.getSymVisibility()) {
        funcOp.setSymVisibilityAttr(tvmffiFuncOp.getSymVisibilityAttr());
      }
      mlir::IRMapping mapping;
      tvmffiFuncOp.getBody().cloneInto(&funcOp.getBody(), mapping);
      const mlir::WalkResult returnConversionResult =
          funcOp.walk([&](tvm_ffi::ReturnOp returnOp) -> mlir::WalkResult {
            if (returnOp.getNumOperands() != targetType.getNumResults()) {
              returnOp.emitError("return value count does not match the "
                                 "converted function signature");
              return mlir::WalkResult::interrupt();
            }
            builder.setInsertionPoint(returnOp);
            mlir::func::ReturnOp::create(builder, returnOp.getLoc(),
                                         returnOp.getOperands());
            returnOp.erase();
            return mlir::WalkResult::advance();
          });
      if (returnConversionResult.wasInterrupted()) {
        signalPassFailure();
        return mlir::WalkResult::interrupt();
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
            tvm_ffi::TVMFFIABIType::getLLVMType(context);
        mlir::LLVM::LLVMPointerType llvmPtrTy =
            mlir::LLVM::LLVMPointerType::get(context);

        builder.setInsertionPointToStart(entry);
        llvm::SmallVector<mlir::Value> arguments;
        mlir::Value allTypesMatch = mlir::arith::ConstantOp::create(
            builder, tvmffiFuncOp.getLoc(), builder.getI1Type(),
            builder.getBoolAttr(true));
        for (auto [index, type] :
             llvm::enumerate(tvmffiFuncOp.getArgumentTypes())) {
          llvm::SmallVector<int32_t> const expected =
              getExpectedTypeIndices(type);
          if (expected.empty()) {
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
          if (expected.size() == 1) {
            condition = mlir::LLVM::ICmpOp::create(
                builder, tvmffiFuncOp.getLoc(), mlir::LLVM::ICmpPredicate::eq,
                actual,
                mlir::LLVM::ConstantOp::create(builder, tvmffiFuncOp.getLoc(),
                                               i32Ty, expected.front()));
          } else {
            condition = mlir::arith::ConstantOp::create(
                builder, tvmffiFuncOp.getLoc(), builder.getI1Type(),
                builder.getBoolAttr(false));
            for (const int32_t expectedIndex : expected) {
              mlir::Value const matches = mlir::LLVM::ICmpOp::create(
                  builder, tvmffiFuncOp.getLoc(), mlir::LLVM::ICmpPredicate::eq,
                  actual,
                  mlir::LLVM::ConstantOp::create(builder, tvmffiFuncOp.getLoc(),
                                                 i32Ty, expectedIndex));
              condition = mlir::arith::OrIOp::create(
                  builder, tvmffiFuncOp.getLoc(), condition, matches);
            }
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
              llvm::enumerate(results), [&](auto indexedResult) -> mlir::Value {
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
        const mlir::Value kindPtr =
            utils::getString(builder, tvmffiFuncOp.getLoc(), "GuardMatch");
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

} // namespace trident::conversion
