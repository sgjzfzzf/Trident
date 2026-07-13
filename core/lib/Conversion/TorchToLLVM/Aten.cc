//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "trident/core/Conversion/TorchToLLVM/TorchToLLVM.h"
#include "trident/core/Conversion/Utils/Check.h"
#include "trident/core/Conversion/Utils/TVMFFIUtils.h"
#include "trident/core/Conversion/Utils/Type.h"
#include "tvm/ffi/c_api.h"
#include "llvm/Support/FormatVariadic.h"
#include <string>

namespace trident::torch {
namespace {

/// Converts torch.aten.* ops by delegating to CAPI, which calls the
/// AtenGen-generated TVM FFI global function wrappers. The op name prefix
/// "torch.aten." is replaced with "trident.aten." to form the FFI function
/// name "trident.aten.<op>.<overload>".
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
    mlir::Location loc = op->getLoc();
    mlir::MLIRContext *ctx = op->getContext();

    // Extract the op name.
    llvm::StringRef opName = op->getName().getStringRef();

    // Expect "torch.aten.<OpName>[.<Overload>]" and map to
    // "trident.aten.<OpName>[.<Overload>]" by replacing the prefix.
    if (!opName.starts_with("torch.aten.")) {
      return mlir::failure();
    }
    std::string globalFuncName =
        llvm::formatv("trident.{0}", opName.drop_front(/*"torch."*/ 6)).str();

    // Find the parent module.
    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return op->emitError("op is not inside a module");
    }

    const size_t numInputs = op->getNumOperands();
    const size_t numResults = op->getNumResults();

    mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
    mlir::LLVM::LLVMStructType anyTy =
        trident::conversion::utils::getTVMFFIAnyType(ctx);

    // Allocate a contiguous TVMFFIAny[] array for the operands on the stack.
    mlir::Value args = mlir::LLVM::AllocaOp::create(
        rewriter, loc, ptrTy, anyTy,
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, numInputs));

    // Store each operand (already a TVMFFIAny struct) into the args array.
    for (auto [i, operand] : llvm::enumerate(operands)) {
      mlir::Value slot =
          mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, anyTy, args,
                                    llvm::ArrayRef<mlir::LLVM::GEPArg>{i});
      mlir::LLVM::StoreOp::create(rewriter, loc, operand, slot);
    }

    // Call the AtenGen-generated FFI wrapper.
    mlir::Value resultSlot = TRIDENT_CHECK_FAILURE(
        trident::conversion::utils::callTVMFFIGlobalFunction(
            rewriter, loc, moduleOp, globalFuncName,
            llvm::map_to_vector(llvm::seq(numInputs),
                                [&](size_t i) -> mlir::Value {
                                  return mlir::LLVM::GEPOp::create(
                                      rewriter, loc, ptrTy, anyTy, args,
                                      llvm::ArrayRef<mlir::LLVM::GEPArg>{i});
                                })));

    // Read the result TVMFFIAny.
    if (numResults == 1) {
      // Single result: load the TVMFFIAny directly from the result slot.
      rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, anyTy, resultSlot);
    } else {
      // Multi-result: AtenGen packs results as a Tuple which maps to
      // a TVMFFIAny with type_index == kTVMFFIArray and payload pointing
      // to an ffi.Array. Extract each element via ffi.ArrayGetItem.
      mlir::Value resultVObj = mlir::LLVM::LoadOp::create(
          rewriter, loc, i64Ty,
          mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, anyTy, resultSlot,
                                    llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2}));

      // Pre-build the array argument slot: TVMFFIAny(kTVMFFIArray, 0, vObj).
      mlir::Value arrayArgSlot = mlir::LLVM::AllocaOp::create(
          rewriter, loc, ptrTy, anyTy,
          mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1));
      mlir::Value kArrayIdx =
          mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, kTVMFFIArray);
      mlir::Value zero32 =
          mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, 0);
      // Build TVMFFIAny {kTVMFFIArray, 0, resultVObj} and store to slot.
      mlir::Value arrayAny = mlir::LLVM::UndefOp::create(rewriter, loc, anyTy);
      arrayAny = mlir::LLVM::InsertValueOp::create(rewriter, loc, anyTy,
                                                   arrayAny, kArrayIdx,
                                                   llvm::ArrayRef<int64_t>{0});
      arrayAny = mlir::LLVM::InsertValueOp::create(
          rewriter, loc, anyTy, arrayAny, zero32, llvm::ArrayRef<int64_t>{1});
      arrayAny = mlir::LLVM::InsertValueOp::create(rewriter, loc, anyTy,
                                                   arrayAny, resultVObj,
                                                   llvm::ArrayRef<int64_t>{2});
      mlir::LLVM::StoreOp::create(rewriter, loc, arrayAny, arrayArgSlot);

      // For each result, call ffi.ArrayGetItem(arrayArgSlot, i) -> TVMFFIAny.
      llvm::SmallVector<mlir::Value> results = llvm::map_to_vector(
          llvm::seq(numResults), [&](size_t i) { // Build index argument slot:
                                                 // TVMFFIAny(kTVMFFIInt, 0, i).
            mlir::Value slot = mlir::LLVM::AllocaOp::create(
                rewriter, loc, ptrTy, anyTy,
                mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1));
            mlir::Value idxI64 =
                mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, i);
            mlir::Value kIntIdx = mlir::LLVM::ConstantOp::create(
                rewriter, loc, i32Ty, kTVMFFIInt);
            mlir::Value idxAny =
                mlir::LLVM::UndefOp::create(rewriter, loc, anyTy);
            idxAny = mlir::LLVM::InsertValueOp::create(
                rewriter, loc, anyTy, idxAny, kIntIdx,
                llvm::ArrayRef<int64_t>{0});
            idxAny = mlir::LLVM::InsertValueOp::create(
                rewriter, loc, anyTy, idxAny, zero32,
                llvm::ArrayRef<int64_t>{1});
            idxAny = mlir::LLVM::InsertValueOp::create(
                rewriter, loc, anyTy, idxAny, idxI64,
                llvm::ArrayRef<int64_t>{2});
            mlir::LLVM::StoreOp::create(rewriter, loc, idxAny, slot);

            // Call ffi.ArrayGetItem(array, index).
            mlir::FailureOr<mlir::Value> elem =
                trident::conversion::utils::callTVMFFIGlobalFunction(
                    rewriter, loc, moduleOp, "ffi.ArrayGetItem",
                    llvm::SmallVector<mlir::Value>{arrayArgSlot, slot});
            mlir::Value result =
                mlir::LLVM::LoadOp::create(rewriter, loc, anyTy, *elem);
            return result;
          });
      rewriter.replaceOp(op, results);
    }
    return mlir::success();
  }
};

} // namespace

void populateTorchToLLVMAtenConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertAtenDispatcherOp>(typeConverter, patterns.getContext());
  // Force torch.aten.* ops through this file's dispatcher lowering, while
  // keeping other Torch dialect ops dynamically legal in this stage.
  target.addDynamicallyLegalDialect<mlir::torch::Torch::TorchDialect>(
      [](mlir::Operation *op) {
        return !op->getName().getStringRef().starts_with("torch.aten.");
      });
}

} // namespace trident::torch
