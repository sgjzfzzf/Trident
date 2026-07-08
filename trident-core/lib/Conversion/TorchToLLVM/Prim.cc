//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "trident-core/Conversion/TorchToLLVM/TorchToLLVM.h"
#include "trident-core/Conversion/Utils/Check.h"
#include "trident-core/Conversion/Utils/TVMFFIUtils.h"
#include "trident-core/Conversion/Utils/Type.h"
#include "tvm/ffi/c_api.h"
#include "llvm/ADT/SmallVectorExtras.h"

namespace trident::torch {
namespace {

/// Converts torch.prim.ListConstruct.
///
/// Constructs an ffi.Array via callTVMFFIGlobalFunction with all elements
/// passed as packed args.  The result is a TVMFFIObjectHandle (!llvm.ptr)
/// which is later converted to a StableListHandle in CAPI when the
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
        trident::conversion::utils::getTVMFFIAnyType(ctx);
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

    // Call ffi.Array(elem0, ..., elemN) - pass each slot individually.
    llvm::SmallVector<mlir::Value> slotPtrs =
        llvm::map_to_vector(llvm::seq(N), [&](size_t i) -> mlir::Value {
          return mlir::LLVM::GEPOp::create(
              rewriter, loc, ptrTy, anyTy, ffiArgs,
              llvm::ArrayRef<mlir::LLVM::GEPArg>{i});
        });
    mlir::Value result = TRIDENT_CHECK_FAILURE(
        trident::conversion::utils::callTVMFFIGlobalFunction(
            rewriter, loc, moduleOp, "ffi.Array", slotPtrs));

    // Extract v_obj (field[2]) from result TVMFFIAny and wrap it back
    // in a TVMFFIAny with kTVMFFIArray tag so downstream consumers
    // (CAPI for aten ops, ConvertObjectDecRefOp, etc.) always
    // see a proper TVMFFIAny value instead of a raw pointer that would
    // force an unreconcilable unrealized_conversion_cast.
    mlir::Value resultSlot = result;
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

} // namespace

void populateTorchToLLVMPrimConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertPrimListConstructOp>(typeConverter,
                                           patterns.getContext());
  target.addIllegalOp<mlir::torch::Torch::PrimListConstructOp>();
}

} // namespace trident::torch
