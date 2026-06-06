// RUN: libtriton-core-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: tvm_ffi.func @test
// CHECK:       tvm_ffi.return
tvm_ffi.func @test() {
  tvm_ffi.return
}

// -----

// CHECK: tvm_ffi.func @with_torch_int
// CHECK: tvm_ffi.return {{%.*}} : !torch.int
tvm_ffi.func @with_torch_int(%arg0: !torch.int) -> !torch.int {
  tvm_ffi.return %arg0 : !torch.int
}
