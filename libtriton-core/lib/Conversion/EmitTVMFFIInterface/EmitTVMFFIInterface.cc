#include <cstdint>

#include "libtriton-core/Analysis/MemRefOriginAnalysis/MemRefOriginAnalysis.h"
#include "libtriton-core/Conversion/EmitTVMFFIInterface/EmitTVMFFIInterface.h"
#include "libtriton-core/Conversion/TVMFFIToLLVM/TVMFFILLVMDescriptors.h"
#include "libtriton-core/Dialect/DLPack/IR/DLPackDialect.h"
#include "libtriton-core/Dialect/DLPack/IR/DLPackOps.h"
#include "libtriton-core/Dialect/DLPack/IR/DLPackTypes.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

namespace libtriton::tvm_ffi {

#define GEN_PASS_DEF_EMITTVMFFIINTERFACE
#include "libtriton-core/Conversion/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kTVMFFIInterfaceAttr =
    "tvm_ffi.emit_tvm_ffi_interface";
constexpr llvm::StringLiteral kTVMFFIWrapperPrefix = "__tvm_ffi_";

mlir::Value emitI32Constant(mlir::OpBuilder &builder, mlir::Location loc,
                            std::int32_t value) {
  return mlir::arith::ConstantIntOp::create(builder, loc, value, 32)
      .getResult();
}

mlir::Value emitI64Constant(mlir::OpBuilder &builder, mlir::Location loc,
                            std::int64_t value) {
  return mlir::arith::ConstantIntOp::create(builder, loc, value, 64)
      .getResult();
}

mlir::LLVM::LLVMPointerType getOpaquePtrType(mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMPointerType::get(context);
}

mlir::Value emitAnyBufferSlotPtr(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::Value bufferPtr,
                                 mlir::TypedValue<mlir::IntegerType> index,
                                 mlir::Type anyType) {
  mlir::Type ptrTy = getOpaquePtrType(builder.getContext());
  return mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyType, bufferPtr,
                                   llvm::ArrayRef<mlir::LLVM::GEPArg>{index})
      .getResult();
}

mlir::FailureOr<mlir::Value> emitUnboxAnyValue(mlir::OpBuilder &builder,
                                               mlir::Location loc,
                                               mlir::Type valueType,
                                               mlir::Value anyValue) {
  mlir::MLIRContext *context = builder.getContext();
  if (mlir::isa<libtriton::tvm_ffi::AnyType>(valueType))
    return anyValue;
  if (mlir::isa<libtriton::tvm_ffi::ObjectHandleType>(valueType)) {
    return libtriton::tvm_ffi::ToObjectOp::create(builder, loc, valueType,
                                                  anyValue)
        .getOutput();
  }
  if (mlir::isa<libtriton::dlpack::DLTensorType>(valueType)) {
    return libtriton::tvm_ffi::ToTensorOp::create(builder, loc, valueType,
                                                  anyValue)
        .getOutput();
  }
  if (mlir::isa<mlir::BaseMemRefType>(valueType)) {
    mlir::Type tensorType = libtriton::dlpack::DLTensorType::get(context);
    mlir::Value tensor = libtriton::tvm_ffi::ToTensorOp::create(
                             builder, loc, tensorType, anyValue)
                             .getOutput();
    return libtriton::dlpack::ToMemRefOp::create(builder, loc, valueType,
                                                 tensor)
        .getOutput();
  }
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(valueType)) {
    return libtriton::tvm_ffi::ToStrOp::create(builder, loc, valueType,
                                               anyValue)
        .getOutput();
  }
  if (mlir::isa<mlir::Float64Type>(valueType)) {
    return libtriton::tvm_ffi::ToFloatOp::create(builder, loc, valueType,
                                                 anyValue)
        .getOutput();
  }
  mlir::IntegerType integerType = mlir::dyn_cast<mlir::IntegerType>(valueType);
  if (integerType && integerType.getWidth() == 64) {
    return libtriton::tvm_ffi::ToIntOp::create(builder, loc, valueType,
                                               anyValue)
        .getOutput();
  }
  mlir::emitError(loc) << "unsupported TVM FFI wrapper parameter type: "
                       << valueType;
  return mlir::failure();
}

mlir::FailureOr<mlir::Value> emitBoxAnyValue(mlir::OpBuilder &builder,
                                             mlir::Location loc,
                                             mlir::DataFlowSolver &solver,
                                             mlir::Value value) {
  mlir::MLIRContext *context = builder.getContext();
  mlir::Type valueType = value.getType();
  mlir::Type anyType = libtriton::tvm_ffi::AnyType::get(context);
  if (mlir::isa<libtriton::tvm_ffi::AnyType>(valueType))
    return value;
  if (mlir::isa<libtriton::tvm_ffi::ObjectHandleType>(valueType)) {
    return libtriton::tvm_ffi::FromObjectOp::create(builder, loc, anyType,
                                                    value)
        .getOutput();
  }
  if (mlir::isa<libtriton::dlpack::DLTensorType>(valueType)) {
    return libtriton::tvm_ffi::FromTensorOp::create(builder, loc, anyType,
                                                    value)
        .getOutput();
  }
  if (mlir::isa<mlir::BaseMemRefType>(valueType)) {
    mlir::Type managedType =
        libtriton::dlpack::DLManagedTensorType::get(context);
    mlir::Type objectHandleType =
        libtriton::tvm_ffi::ObjectHandleType::get(context);
    const libtriton::analysis::MemRefOriginKind origin =
        libtriton::analysis::resolveMemRefOrigin(solver, value);

    // TVMFFITensorFromDLPack currently accepts !dlpack.managed_tensor, so use
    // FromMemRefOwnedOp for all origins and keep the origin query for future
    // policy extension.
    (void)origin;
    mlir::Value managed = libtriton::dlpack::FromMemRefOwnedOp::create(
                              builder, loc, managedType, value)
                              .getOutput();

    mlir::Value zero =
        mlir::arith::ConstantIntOp::create(builder, loc, 0, 32).getResult();
    mlir::Value handle =
        libtriton::tvm_ffi::TensorFromDLPackOp::create(
            builder, loc, objectHandleType, managed, zero, zero)
            .getOutput();
    return libtriton::tvm_ffi::FromTensorOp::create(builder, loc, anyType,
                                                    handle)
        .getOutput();
  }
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(valueType)) {
    return libtriton::tvm_ffi::FromStrOp::create(builder, loc, anyType, value)
        .getOutput();
  }
  if (mlir::isa<mlir::Float64Type>(valueType)) {
    return libtriton::tvm_ffi::FromFloatOp::create(builder, loc, anyType, value)
        .getOutput();
  }
  mlir::IntegerType integerType = mlir::dyn_cast<mlir::IntegerType>(valueType);
  if (integerType && integerType.getWidth() == 64) {
    return libtriton::tvm_ffi::FromIntOp::create(builder, loc, anyType, value)
        .getOutput();
  }
  mlir::emitError(loc) << "unsupported TVM FFI wrapper return type: "
                       << valueType;
  return mlir::failure();
}

mlir::FailureOr<mlir::func::FuncOp>
buildEmitTVMFFIInterfaceWrapper(mlir::ModuleOp moduleOp,
                                mlir::DataFlowSolver &solver,
                                mlir::func::FuncOp targetFunc) {
  mlir::MLIRContext *context = moduleOp.getContext();
  mlir::Location loc = targetFunc.getLoc();
  mlir::Type ptrTy = getOpaquePtrType(context);
  mlir::Type i32Ty = mlir::IntegerType::get(context, 32);
  mlir::Type anyTy = libtriton::tvm_ffi::AnyType::get(context);
  mlir::Type anyLLVMType =
      libtriton::conversion::utils::TVMFFIAnyLLVMDescriptor::getLLVMType(
          context);

  const std::string wrapperName =
      (kTVMFFIWrapperPrefix + targetFunc.getSymName()).str();
  if (mlir::SymbolTable::lookupSymbolIn(moduleOp, wrapperName)) {
    targetFunc.emitError() << "duplicate wrapper symbol " << wrapperName;
    return mlir::failure();
  }

  mlir::FunctionType targetType = targetFunc.getFunctionType();
  if (targetType.getNumResults() > 1) {
    targetFunc.emitError()
        << "tvm_ffi.emit_tvm_ffi_interface only supports at most one return "
           "value";
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Type> wrapperInputs = {ptrTy, ptrTy, i32Ty, ptrTy};
  llvm::SmallVector<mlir::Type> wrapperResults = {i32Ty};
  mlir::FunctionType wrapperType =
      mlir::FunctionType::get(context, wrapperInputs, wrapperResults);

  mlir::OpBuilder moduleBuilder(context);
  moduleBuilder.setInsertionPointToEnd(moduleOp.getBody());
  mlir::func::FuncOp wrapperFunc =
      mlir::func::FuncOp::create(loc, wrapperName, wrapperType);
  moduleBuilder.insert(wrapperFunc);

  mlir::Block *entryBlock = wrapperFunc.addEntryBlock();
  mlir::OpBuilder invokeBuilder = mlir::OpBuilder::atBlockEnd(entryBlock);
  llvm::SmallVector<mlir::Value> callArgs;
  callArgs.reserve(targetType.getNumInputs());
  mlir::Value packedArgsPtr = entryBlock->getArgument(1);
  mlir::Value packedResultPtr = entryBlock->getArgument(3);
  for (std::int32_t i = 0; i < targetType.getNumInputs(); ++i) {
    mlir::TypedValue<mlir::IntegerType> argIndex =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            emitI64Constant(invokeBuilder, loc, i));
    mlir::Value argSlotPtr = emitAnyBufferSlotPtr(
        invokeBuilder, loc, packedArgsPtr, argIndex, anyLLVMType);
    mlir::Value anyValueLLVM =
        mlir::LLVM::LoadOp::create(invokeBuilder, loc, anyLLVMType, argSlotPtr)
            .getResult();
    mlir::Value anyValue = libtriton::tvm_ffi::AnyFromLLVMOp::create(
                               invokeBuilder, loc, anyTy, anyValueLLVM)
                               .getOutput();
    mlir::FailureOr<mlir::Value> unpackedValue =
        emitUnboxAnyValue(invokeBuilder, loc, targetType.getInput(i), anyValue);
    if (mlir::failed(unpackedValue))
      return mlir::failure();
    callArgs.push_back(*unpackedValue);
  }

  mlir::func::CallOp callOp =
      mlir::func::CallOp::create(invokeBuilder, loc, targetFunc, callArgs);
  if (callOp.getNumResults() == 1) {
    mlir::FailureOr<mlir::Value> boxedResult =
        emitBoxAnyValue(invokeBuilder, loc, solver, callOp.getResult(0));
    if (mlir::failed(boxedResult))
      return mlir::failure();
    mlir::TypedValue<mlir::IntegerType> resultIndex =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            emitI64Constant(invokeBuilder, loc, 0));
    mlir::Value resultSlotPtr = emitAnyBufferSlotPtr(
        invokeBuilder, loc, packedResultPtr, resultIndex, anyLLVMType);
    mlir::Value boxedResultLLVM =
        libtriton::tvm_ffi::AnyToLLVMOp::create(invokeBuilder, loc, anyLLVMType,
                                                *boxedResult)
            .getOutput();
    mlir::LLVM::StoreOp::create(invokeBuilder, loc, boxedResultLLVM,
                                resultSlotPtr);
  }
  mlir::Value successCode = emitI32Constant(invokeBuilder, loc, 0);
  mlir::func::ReturnOp::create(invokeBuilder, loc, successCode);
  return wrapperFunc;
}

class EmitTVMFFIInterfacePass
    : public impl::EmitTVMFFIInterfaceBase<EmitTVMFFIInterfacePass> {
public:
  void runOnOperation() final {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::DataFlowSolver solver;
    mlir::dataflow::loadBaselineAnalyses(solver);
    solver.load<libtriton::analysis::MemRefOriginDataFlowAnalysis>();
    if (mlir::failed(solver.initializeAndRun(moduleOp))) {
      signalPassFailure();
      return;
    }

    for (mlir::func::FuncOp targetFunc :
         llvm::make_filter_range(moduleOp.getOps<mlir::func::FuncOp>(),
                                 [](mlir::func::FuncOp funcOp) {
                                   return !funcOp.isDeclaration() &&
                                          funcOp->hasAttr(kTVMFFIInterfaceAttr);
                                 })) {
      if (mlir::failed(
              buildEmitTVMFFIInterfaceWrapper(moduleOp, solver, targetFunc))) {
        signalPassFailure();
        return;
      }
      targetFunc->removeAttr(kTVMFFIInterfaceAttr);
    }
  }
};

} // namespace

} // namespace libtriton::tvm_ffi
