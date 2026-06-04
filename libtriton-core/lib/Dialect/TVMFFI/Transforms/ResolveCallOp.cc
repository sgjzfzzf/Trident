//===- ResolveCallOp.cc ---------------------------------------------------===//
//  The `finalize-tvm-ffi-call` pass implementation.
//
//  Resolves each `tvm_ffi.call "name"(args)` by:
//  1. Collecting all unique function names used across the module.
//  2. Creating a module-level `llvm.mlir.global` of type `!llvm.ptr` for
//     each unique name to cache the resolved function handle.
//  3. Creating a single `llvm.func` init function that calls
//     `tvm_ffi.function_get_global` for every name, converts each result
//     to `!llvm.ptr` and stores it into the corresponding global.
//  4. Creating a single `llvm.func` fini function that loads each handle
//     from its global, converts back to `!tvm_ffi.object_handle` and calls
//     `tvm_ffi.object_dec_ref`.
//  5. Registering the init / fini via `llvm.mlir.global_ctors` /
//  `global_dtors`.
//  6. Replacing each `tvm_ffi.call` with a load from its global + a
//     `tvm_ffi.function_call` using the loaded handle.
//===----------------------------------------------------------------------===//

#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "libtriton-core/Dialect/TVMFFI/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"

namespace libtriton::tvm_ffi {

#define GEN_PASS_DEF_FINALIZETVMFFICALL
#include "libtriton-core/Dialect/TVMFFI/Transforms/Passes.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string getGlobalSymName(llvm::StringRef funcName) {
  return llvm::formatv("__tvm_ffi_{0}", funcName);
}

/// Create the single init function that resolves ALL function handles.
static mlir::LLVM::LLVMFuncOp
createGlobalInitFunction(mlir::OpBuilder &builder, mlir::Location loc,
                         llvm::StringRef symName,
                         llvm::ArrayRef<llvm::StringRef> funcNames) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::LLVM::LLVMFunctionType funcType = mlir::LLVM::LLVMFunctionType::get(
      mlir::LLVM::LLVMVoidType::get(ctx), /*params=*/{});

  mlir::LLVM::LLVMFuncOp funcOp = mlir::LLVM::LLVMFuncOp::create(
      builder, loc, symName, funcType, mlir::LLVM::linkage::Linkage::Internal);

  mlir::Block *entry = funcOp.addEntryBlock(builder);
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);

  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::Type handleTy = ObjectHandleType::get(ctx);

  for (llvm::StringRef funcName : funcNames) {
    std::string globalSymName = getGlobalSymName(funcName);
    FunctionGetGlobalOp getGlobalOp =
        FunctionGetGlobalOp::create(builder, loc, handleTy, funcName);
    mlir::LLVM::AddressOfOp addr =
        mlir::LLVM::AddressOfOp::create(builder, loc, ptrTy, globalSymName);
    StoreOp::create(builder, loc, getGlobalOp.getResult(), addr);
  }

  mlir::LLVM::ReturnOp::create(builder, loc, mlir::ValueRange{});
  return funcOp;
}

/// Create the single fini function that destroys ALL function handles.
static mlir::LLVM::LLVMFuncOp
createGlobalFiniFunction(mlir::OpBuilder &builder, mlir::Location loc,
                         llvm::StringRef symName,
                         llvm::ArrayRef<llvm::StringRef> funcNames) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::LLVM::LLVMFunctionType funcType = mlir::LLVM::LLVMFunctionType::get(
      mlir::LLVM::LLVMVoidType::get(ctx), /*params=*/{});

  mlir::LLVM::LLVMFuncOp funcOp = mlir::LLVM::LLVMFuncOp::create(
      builder, loc, symName, funcType, mlir::LLVM::linkage::Linkage::Internal);

  mlir::Block *entry = funcOp.addEntryBlock(builder);
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);

  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::Type handleTy = ObjectHandleType::get(ctx);

  for (llvm::StringRef funcName : funcNames) {
    std::string globalSymName = getGlobalSymName(funcName);
    mlir::LLVM::AddressOfOp addr =
        mlir::LLVM::AddressOfOp::create(builder, loc, ptrTy, globalSymName);
    LoadOp loadOp = LoadOp::create(builder, loc, handleTy, addr);
    ObjectDecRefOp::create(builder, loc, loadOp.getResult());
  }

  mlir::LLVM::ReturnOp::create(builder, loc, mlir::ValueRange{});
  return funcOp;
}

/// Rewrite pattern that replaces `tvm_ffi.call` with a load from the
/// cached global handle followed by a `tvm_ffi.function_call`.
struct CallOpPattern : public mlir::OpRewritePattern<CallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(CallOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = op.getContext();
    mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::Type handleTy = ObjectHandleType::get(ctx);
    std::string globalSymName = getGlobalSymName(op.getFuncName());

    mlir::LLVM::AddressOfOp addrOp =
        mlir::LLVM::AddressOfOp::create(rewriter, loc, ptrTy, globalSymName);
    LoadOp loadOp = LoadOp::create(rewriter, loc, handleTy, addrOp);
    rewriter.replaceOpWithNewOp<FunctionCallOp>(
        op, op.getResult().getType(), loadOp.getResult(), op.getArgs());
    return mlir::success();
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Pass implementation
// ---------------------------------------------------------------------------

class FinalizeTVMFFICallPass
    : public impl::FinalizeTVMFFICallBase<FinalizeTVMFFICallPass> {
public:
  FinalizeTVMFFICallPass() = default;

  FinalizeTVMFFICallPass(const FinalizeTVMFFICallOptions &opts)
      : FinalizeTVMFFICallBase(opts) {}

  void runOnOperation() final {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::MLIRContext *ctx = &getContext();
    mlir::OpBuilder builder(ctx);

    // Collect unique function names in insertion order.
    llvm::SetVector<llvm::StringRef> funcNames;
    moduleOp.walk([&](CallOp op) { funcNames.insert(op.getFuncName()); });

    if (funcNames.empty()) {
      return;
    }

    // Create all global variables at the top of the module.
    mlir::Location loc = moduleOp.getLoc();
    mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    builder.setInsertionPointToStart(moduleOp.getBody());
    for (llvm::StringRef funcName : funcNames) {
      mlir::LLVM::GlobalOp::create(builder, loc, ptrTy, /*isConstant=*/false,
                                   mlir::LLVM::linkage::Linkage::Internal,
                                   getGlobalSymName(funcName),
                                   /*value=*/mlir::Attribute{});
    }

    // Create the single init function that resolves all handles.
    mlir::LLVM::LLVMFuncOp initFn = createGlobalInitFunction(
        builder, loc, initName, funcNames.getArrayRef());

    // Create the single fini function that destroys all handles.
    mlir::LLVM::LLVMFuncOp finiFn = createGlobalFiniFunction(
        builder, loc, finiName, funcNames.getArrayRef());

    // Register via single llvm.mlir.global_ctors / global_dtors.
    mlir::LLVM::ZeroAttr zeroData = mlir::LLVM::ZeroAttr::get(ctx);
    mlir::SymbolRefAttr ctorRef =
        mlir::SymbolRefAttr::get(ctx, initFn.getName());
    mlir::LLVM::GlobalCtorsOp::create(
        builder, loc, builder.getArrayAttr({ctorRef}),
        builder.getArrayAttr({builder.getI32IntegerAttr(0)}),
        builder.getArrayAttr({zeroData}));
    mlir::SymbolRefAttr dtorRef =
        mlir::SymbolRefAttr::get(ctx, finiFn.getName());
    mlir::LLVM::GlobalDtorsOp::create(
        builder, loc, builder.getArrayAttr({dtorRef}),
        builder.getArrayAttr({builder.getI32IntegerAttr(0)}),
        builder.getArrayAttr({zeroData}));

    // Rewrite each CallOp using a pattern.
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<CallOpPattern>(ctx);
    if (mlir::failed(
            mlir::applyPatternsGreedily(moduleOp, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace libtriton::tvm_ffi
