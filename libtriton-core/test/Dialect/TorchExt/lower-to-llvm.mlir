// RUN: libtriton-core-opt %s -convert-to-llvm | FileCheck %s

// CHECK-DAG: llvm.func @__libtriton_get_current_device() -> i8
// CHECK-DAG: llvm.func @__libtriton_get_current_stream(i8) -> !llvm.ptr

// CHECK-LABEL: func.func @lowering_get_current_device
// CHECK-SAME: () -> i8
func.func @lowering_get_current_device() -> i8 {
  // CHECK-NOT: torch_ext.get_current_device
  // CHECK: %[[DEV:.*]] = llvm.call @__libtriton_get_current_device() : () -> i8
  // CHECK: return %[[DEV]] : i8
  %0 = torch_ext.get_current_device : i8
  return %0 : i8
}

// CHECK-LABEL: func.func @lowering_get_current_stream_with_device
// CHECK-SAME: (%[[DEVICE:.*]]: i8) -> !llvm.ptr
func.func @lowering_get_current_stream_with_device(%device: i8) -> !llvm.ptr {
  // CHECK-NOT: torch_ext.get_current_stream
  // CHECK: %[[STREAM:.*]] = llvm.call @__libtriton_get_current_stream(%[[DEVICE]]) : (i8) -> !llvm.ptr
  // CHECK: return %[[STREAM]] : !llvm.ptr
  %0 = torch_ext.get_current_stream device %device : i8 : !llvm.ptr
  return %0 : !llvm.ptr
}

// CHECK-LABEL: func.func @lowering_get_current_stream_default_device
// CHECK-SAME: () -> !llvm.ptr
func.func @lowering_get_current_stream_default_device() -> !llvm.ptr {
  // CHECK-NOT: torch_ext.get_current_stream
  // CHECK: %[[DEFAULT_DEVICE:.*]] = llvm.mlir.constant(-1 : i8) : i8
  // CHECK: %[[STREAM:.*]] = llvm.call @__libtriton_get_current_stream(%[[DEFAULT_DEVICE]]) : (i8) -> !llvm.ptr
  // CHECK: return %[[STREAM]] : !llvm.ptr
  %0 = torch_ext.get_current_stream : !llvm.ptr
  return %0 : !llvm.ptr
}
