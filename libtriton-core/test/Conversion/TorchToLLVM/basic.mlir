// RUN: libtriton-core-opt %s --convert-torch-to-llvm | FileCheck %s
//
// Tests that the standalone ConvertTorchToLLVM pass lowers
// torch.constant.bool/int/float to LLVM::ConstantOp while leaving
// function signatures unchanged (type conversion is handled by the
// separate func-backend-type-conversion pass).
//
// torch.constant.none has no dedicated pattern and is handled by a
// generic materialization (NoneType -> i64).

// CHECK-LABEL:   func.func @torch.constant.bool() -> !torch.bool {
// CHECK:           %[[TRUE:.*]] = llvm.mlir.constant(true) : i1
// CHECK:           %[[C:.*]] = builtin.unrealized_conversion_cast %[[TRUE]] : i1 to !torch.bool
// CHECK-NEXT:      return %[[C]] : !torch.bool
func.func @torch.constant.bool() -> !torch.bool {
  %true = torch.constant.bool true
  return %true : !torch.bool
}

// CHECK-LABEL:   func.func @torch.constant.int() -> !torch.int {
// CHECK:           %[[C42:.*]] = llvm.mlir.constant(42 : i64) : i64
// CHECK:           %[[C:.*]] = builtin.unrealized_conversion_cast %[[C42]] : i64 to !torch.int
// CHECK-NEXT:      return %[[C]] : !torch.int
func.func @torch.constant.int() -> !torch.int {
  %int = torch.constant.int 42
  return %int : !torch.int
}

// CHECK-LABEL:   func.func @torch.constant.float() -> !torch.float {
// CHECK:           %[[C3_14:.*]] = llvm.mlir.constant(3.140000e+00 : f64) : f64
// CHECK:           %[[C:.*]] = builtin.unrealized_conversion_cast %[[C3_14]] : f64 to !torch.float
// CHECK-NEXT:      return %[[C]] : !torch.float
func.func @torch.constant.float() -> !torch.float {
  %float = torch.constant.float 3.14
  return %float : !torch.float
}

// CHECK-LABEL:   func.func @torch.constant.none() -> !torch.none {
// CHECK:           %[[NONE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[C:.*]] = builtin.unrealized_conversion_cast %[[NONE]] : i64 to !torch.none
// CHECK-NEXT:      return %[[C]] : !torch.none
func.func @torch.constant.none() -> !torch.none {
  %none = torch.constant.none
  return %none : !torch.none
}
