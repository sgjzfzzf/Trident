//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "trident-refcnt"
#include "trident/core/Conversion/TorchToLLVM/TorchToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "trident/core/Conversion/Utils/TVMFFICAPIDescriptors.h"
#include "trident/core/Conversion/Utils/Type.h"
#include "trident/core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

namespace trident::torch {

#define GEN_PASS_DEF_CONVERTTORCHTOLLVM
#include "trident/core/Conversion/Passes.h.inc"

namespace {

/// Command-line switch for the ref-counting debug output.
///
/// This TU is compiled as part of a Debug build (no NDEBUG), so the
/// `LLVM_DEBUG` macro is live.  However the linked LLVM library is a Release
/// build in which the stock `-debug` / `-debug-only` options are not
/// registered, so this self-registered option drives `llvm::DebugFlag`
/// directly (the `DebugFlag` symbol itself is always present in the library);
/// `isCurrentDebugType` returns true when no `-debug-only` filter is set, so
/// all `LLVM_DEBUG` output in this file becomes visible.
static llvm::cl::opt<bool> enableTridentRefCntDebug(
    "trident-refcnt-debug",
    llvm::cl::desc("Print reference-counting decisions made by the "
                   "Torch->LLVM conversion (IncRef/DecRef insertion)."),
    llvm::cl::init(false));

} // namespace

namespace {
static mlir::Value stripUnrealizedCast(mlir::Value value) {
  while (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() != 1)
      break;
    value = cast->getOperand(0);
  }
  return value;
}

/// Get-or-create the TVMFFIObject{Inc,Dec}Ref C API callee declaration.
///
/// The getter is passed as a template parameter so the Inc and Dec variants
/// share one implementation; the call sites below bind it to
/// getOrCreateTVMFFIObject{Inc,Dec}Ref.
template <auto GetCallee>
static mlir::LLVM::LLVMFuncOp getObjectRefCallee(mlir::ModuleOp moduleOp) {
  mlir::FailureOr<mlir::LLVM::LLVMFuncOp> callee = GetCallee(moduleOp);
  assert(mlir::succeeded(callee) &&
         "failed to create TVMFFIObjectIncRef/DecRef declaration");
  return *callee;
}

/// Emit `TVMFFIObject{Inc,Dec}Ref(handle)` for a converted TVMFFIAny value:
///   %payload = llvm.extractvalue %any[2]
///   %handle  = llvm.inttoptr %payload : i64 to !llvm.ptr
///   llvm.call @TVMFFIObject{Inc,Dec}Ref(%handle) : (!llvm.ptr) -> i32
///
/// The getter is passed as a template parameter (bound to the Inc/Dec
/// variants by the wrappers below), so the two call sites are distinct
/// functions instead of sharing a runtime bool dispatch.
template <auto GetCallee>
static void insertObjectRefCall(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::ModuleOp moduleOp, mlir::Value anyValue) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::LLVM::LLVMFuncOp callee = getObjectRefCallee<GetCallee>(moduleOp);
  mlir::Value payload = mlir::LLVM::ExtractValueOp::create(
      builder, loc, anyValue, llvm::ArrayRef<int64_t>{2});
  mlir::Value handle = mlir::LLVM::IntToPtrOp::create(
      builder, loc, mlir::LLVM::LLVMPointerType::get(ctx), payload);
  mlir::LLVM::CallOp::create(builder, loc, callee, mlir::ValueRange{handle});
}

/// Emit `TVMFFIObjectIncRef(handle)` for a converted TVMFFIAny value.
static void insertObjectIncRefCall(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::ModuleOp moduleOp,
                                   mlir::Value anyValue) {
  insertObjectRefCall<
      &trident::conversion::utils::getOrCreateTVMFFIObjectIncRef>(
      builder, loc, moduleOp, anyValue);
}

/// Emit `TVMFFIObjectDecRef(handle)` for a converted TVMFFIAny value.
static void insertObjectDecRefCall(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::ModuleOp moduleOp,
                                   mlir::Value anyValue) {
  insertObjectRefCall<
      &trident::conversion::utils::getOrCreateTVMFFIObjectDecRef>(
      builder, loc, moduleOp, anyValue);
}

class ConvertTorchToLLVMPass
    : public impl::ConvertTorchToLLVMBase<ConvertTorchToLLVMPass> {
public:
  void runOnOperation() final {
    // Enable LLVM_DEBUG output for this file.  The `-debug`/`-debug-only`
    // options are not registered (the linked LLVM library is a Release
    // build), so the self-registered -trident-refcnt-debug option drives the
    // always-present `llvm::DebugFlag` symbol directly.
    llvm::DebugFlag = enableTridentRefCntDebug;

    mlir::LLVMTypeConverter typeConverter(&getContext());
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());

    // Populated during the conversion with the converted object values; the
    // post-conversion ref-counting pass below consumes it.
    RefCountTable refCountTable;
    populateTorchToLLVMConversionPatterns(target, typeConverter, patterns,
                                          refCountTable);

    // Fold only as a fallback *after* running the conversion patterns.
    // With the default `BeforePatterns` mode, torch.aten.clone (whose fold
    // unconditionally returns the self operand when the types match, ignoring
    // memory_format) gets folded before ConvertAtenDispatcherOp can lower it to
    // the trident.aten.clone FFI call. That would alias the clone result with
    // its operand, unbalancing the reference counting inserted by RAAI and
    // causing a premature release / double free at runtime.
    if (mlir::failed(mlir::applyPartialConversion(
            getOperation(), target, std::move(patterns),
            mlir::ConversionConfig{
                .foldingMode =
                    mlir::DialectConversionFoldingMode::AfterPatterns}))) {
      signalPassFailure();
    } else {
      // Insert TVMFFIObjectIncRef/DecRef for the objects produced during the
      // conversion.  The Torch types are gone by now (converted values are
      // TVMFFIAny), so the decision is driven by the table populated above.
      //
      // For every single-block region:
      //   * escaping objects — values used by the terminator operands (they
      //     cross the scope boundary) — get an IncRef;
      //   * objects produced by ops directly in the block get a DecRef (the
      //     last use inside this scope consumes the reference).
      //
      // IncRef/DecRef of the same object emitted back-to-back at the same
      // insertion point cancel out at runtime (net-zero), so the counting stays
      // balanced without a separate pair-elimination pass.
      mlir::ModuleOp moduleOp = getOperation();

      LLVM_DEBUG({
        llvm::dbgs() << "[trident-refcnt] tracking " << refCountTable.size()
                     << " converted object(s)\n";
        for (auto &entry : refCountTable)
          llvm::dbgs() << "[trident-refcnt]   object: " << entry.first
                       << " (net " << entry.second << ")\n";
      });

      // Insertions happen directly inside the walk.  This is safe with
      // Operation::walk: the callback runs in pre-order *before* the walk
      // descends into `op`'s regions, and blocks are iterated with
      // early-increment over a stable ilist — so ops spliced before a block's
      // terminator (or the lazily-created llvm.func declaration at the front
      // of the module body) never invalidate an iterator.  The spliced ops
      // are region-less llvm ops, so the walk simply visits and ignores them.
      moduleOp->walk([&](mlir::Operation *op) {
        for (mlir::Region &region : op->getRegions()) {
          // Empty blocks have no terminator.  Calling getTerminator() on one
          // is UB: in release builds the mightHaveTerminator() assert is
          // compiled out and back() returns the ilist sentinel embedded in the
          // Block object — not a valid Operation* — so reading its location
          // SIGSEGVs.  Guard on both a single-block region and a non-empty
          // block instead.
          if (!llvm::isa<mlir::ModuleOp>(op) && region.hasOneBlock()) {
            mlir::Block &block = region.front();
            mlir::Operation *terminator = block.getTerminator();
            mlir::OpBuilder builder(terminator);
            mlir::Location loc = terminator->getLoc();

            // --- IncRef escaping objects (terminator operands) ---
            for (mlir::Value operand : terminator->getOperands()) {
              mlir::Value llvmValue = stripUnrealizedCast(operand);
              auto it = refCountTable.find(llvmValue);
              if (it != refCountTable.end()) {
                LLVM_DEBUG(llvm::dbgs() << "[trident-refcnt] IncRef " << operand
                                        << " (escapes through terminator of "
                                        << op->getName() << ")\n");
                it->second += 1;
                insertObjectIncRefCall(builder, loc, moduleOp, llvmValue);
              }
              // Pass-through escape: a ref-counted object that is *not*
              // produced by a converted op is invisible to the table — most
              // notably a func block argument returned directly (e.g. a
              // pre-allocated output tensor such as flag_gems' mm).  The FFI
              // runtime releases the returned handle (ownership transfer), so
              // the escaping reference must still be IncRef'd here, mirroring
              // the legacy RAAI which IncRef'd every terminator operand
              // unconditionally.  The value is still Torch-typed at this
              // stage (func signature conversion runs later in the pipeline),
              // so the TVMFFIAny is materialized with an unrealized conversion
              // cast, which the backend type conversion turns into an identity
              // cast that ReconcileUnrealizedCasts folds away.
              else if (llvm::isa<mlir::BlockArgument>(llvmValue) &&
                       isRefCountedObjectType(llvmValue.getType())) {
                LLVM_DEBUG(llvm::dbgs()
                           << "[trident-refcnt] IncRef " << llvmValue
                           << " (block arg escapes through terminator of "
                           << op->getName() << ")\n");
                mlir::Value anyValue =
                    mlir::UnrealizedConversionCastOp::create(
                        builder, loc,
                        trident::conversion::utils::getTVMFFIAnyType(
                            builder.getContext()),
                        llvmValue)
                        .getResult(0);
                insertObjectIncRefCall(builder, loc, moduleOp, anyValue);
              }
            }

            // --- DecRef objects produced by ops directly in this block ---
            // The table keys are the *converted* (remapped) values, so
            // looking up the original op results never matches — iterate the
            // table instead and DecRef every entry whose defining op lives
            // directly in this block (the last use inside this scope consumes
            // the reference).  Objects that also escape through the terminator
            // get an IncRef above; the two cancel out at runtime (net-zero).
            for (auto &entry : refCountTable) {
              mlir::Value value = entry.first;
              mlir::Operation *definingOp = value.getDefiningOp();
              if (definingOp && definingOp->getBlock() == &block) {
                LLVM_DEBUG(llvm::dbgs()
                           << "[trident-refcnt] DecRef " << value
                           << " (produced by " << definingOp->getName()
                           << ", last use inside scope)\n");
                entry.second -= 1;
                insertObjectDecRefCall(builder, loc, moduleOp, value);
              }
            }
          }
        }
      });
    }
  }
};

struct TorchToLLVMDialectInterface
    : public mlir::ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(
      mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
      mlir::RewritePatternSet &patterns) const final {
    // The ConvertToLLVM interface path performs no ref-counting: pass a
    // throwaway table that the patterns fill but nobody consumes.
    RefCountTable refCountTable;
    populateTorchToLLVMConversionPatterns(target, typeConverter, patterns,
                                          refCountTable);
  }
};

} // namespace

void populateTorchToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns, RefCountTable &refCountTable) {
  setupBackendTypeConversion(target, typeConverter);
  populateTorchToLLVMConstantConversionPatterns(target, typeConverter,
                                                patterns);
  populateTorchToLLVMLiteralConversionPatterns(target, typeConverter, patterns);
  populateTorchToLLVMAtenConversionPatterns(target, typeConverter, patterns,
                                            refCountTable);
  populateTorchToLLVMPrimConversionPatterns(target, typeConverter, patterns,
                                            refCountTable);
  target.addLegalDialect<mlir::LLVM::LLVMDialect, mlir::BuiltinDialect,
                         mlir::func::FuncDialect>();
}

void registerConvertTorchToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::torch::Torch::TorchDialect *dialect) {
        dialect->addInterfaces<TorchToLLVMDialectInterface>();
      });
}

} // namespace trident::torch
