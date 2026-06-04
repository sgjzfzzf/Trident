// RUN: libtriton-core-opt %s -split-input-file -finalize-tvm-ffi-call | FileCheck %s --check-prefix=CHECK-TVM-FFI
// RUN: libtriton-core-opt %s -split-input-file -finalize-tvm-ffi-call -convert-tvm-ffi-to-llvm | FileCheck %s --check-prefix=CHECK-LLVM

// -----
// CHECK-TVM-FFI:      llvm.mlir.global internal @__tvm_ffi_my_func

// CHECK-TVM-FFI:      llvm.func {{.*}} @__tvm_ffi_init
// CHECK-TVM-FFI:         [[ADDR:%.*]] = llvm.mlir.addressof @__tvm_ffi_my_func
// CHECK-TVM-FFI:         [[H:%.*]] = tvm_ffi.function_get_global "my_func"
// CHECK-TVM-FFI:         tvm_ffi.store [[H]], [[ADDR]]
// CHECK-TVM-FFI:         llvm.return

// CHECK-TVM-FFI:      llvm.func {{.*}} @__tvm_ffi_fini
// CHECK-TVM-FFI:         [[H2:%.*]] = tvm_ffi.load %{{.*}} : !llvm.ptr -> !tvm_ffi.object_handle
// CHECK-TVM-FFI:         tvm_ffi.object_dec_ref [[H2]]
// CHECK-TVM-FFI:         llvm.return

// CHECK-TVM-FFI:      llvm.mlir.global_ctors
// CHECK-TVM-FFI:      llvm.mlir.global_dtors

// CHECK-LLVM:      llvm.func @TVMFFIFunctionCall
// CHECK-LLVM:      llvm.func @TVMFFIObjectDecRef
// CHECK-LLVM:      llvm.mlir.global internal constant @__libtriton_tvm_ffi_func_name_my_func
// CHECK-LLVM:      llvm.func @TVMFFIFunctionGetGlobal
// CHECK-LLVM:      llvm.mlir.global internal @__tvm_ffi_my_func

// CHECK-LLVM:      llvm.func {{.*}} @__tvm_ffi_init
// CHECK-LLVM:         llvm.mlir.addressof
// CHECK-LLVM:         llvm.call @TVMFFIFunctionGetGlobal

// CHECK-LLVM:      llvm.func {{.*}} @__tvm_ffi_fini
// CHECK-LLVM:         llvm.load
// CHECK-LLVM:         llvm.call @TVMFFIObjectDecRef
// CHECK-LLVM:         llvm.return

// CHECK-LLVM:      llvm.mlir.global_ctors
// CHECK-LLVM:      llvm.mlir.global_dtors

// CHECK-TVM-FFI-LABEL: func.func @call_with_args
func.func @call_with_args(%arg0: !tvm_ffi.any, %arg1: !tvm_ffi.any) -> !tvm_ffi.any {
  // CHECK-TVM-FFI-NOT:  tvm_ffi.call
  // CHECK-TVM-FFI:      [[ADDR:%.*]] = llvm.mlir.addressof @__tvm_ffi_my_func
  // CHECK-TVM-FFI:      [[HANDLE:%.*]] = tvm_ffi.load [[ADDR]]
  // CHECK-TVM-FFI:      tvm_ffi.function_call [[HANDLE]](
  // CHECK-LLVM:      llvm.mlir.addressof
  // CHECK-LLVM:      llvm.call @TVMFFIFunctionCall
  %0 = tvm_ffi.call "my_func"(%arg0, %arg1) : (!tvm_ffi.any, !tvm_ffi.any) -> !tvm_ffi.any
  return %0 : !tvm_ffi.any
}

// -----
// CHECK-TVM-FFI:      llvm.mlir.global internal @__tvm_ffi_alpha
// CHECK-TVM-FFI:      llvm.mlir.global internal @__tvm_ffi_beta

// CHECK-TVM-FFI:      llvm.func {{.*}} @__tvm_ffi_init
// CHECK-TVM-FFI:         [[HA:%.*]] = tvm_ffi.function_get_global "alpha"
// CHECK-TVM-FFI:         [[HB:%.*]] = tvm_ffi.function_get_global "beta"
// CHECK-TVM-FFI:         llvm.return

// CHECK-TVM-FFI:      llvm.func {{.*}} @__tvm_ffi_fini
// CHECK-TVM-FFI:         [[LA:%.*]] = tvm_ffi.load %{{.*}} : !llvm.ptr -> !tvm_ffi.object_handle
// CHECK-TVM-FFI:         tvm_ffi.object_dec_ref [[LA]]
// CHECK-TVM-FFI:         [[LB:%.*]] = tvm_ffi.load %{{.*}} : !llvm.ptr -> !tvm_ffi.object_handle
// CHECK-TVM-FFI:         tvm_ffi.object_dec_ref [[LB]]
// CHECK-TVM-FFI:         llvm.return

// CHECK-LLVM:      llvm.func @TVMFFIFunctionCall
// CHECK-LLVM:      llvm.func @TVMFFIObjectDecRef
// CHECK-LLVM:      llvm.mlir.global internal constant @__libtriton_tvm_ffi_func_name_beta
// CHECK-LLVM:      llvm.mlir.global internal constant @__libtriton_tvm_ffi_func_name_alpha
// CHECK-LLVM:      llvm.func @TVMFFIFunctionGetGlobal
// CHECK-LLVM:      llvm.mlir.global internal @__tvm_ffi_alpha
// CHECK-LLVM:      llvm.mlir.global internal @__tvm_ffi_beta
// CHECK-LLVM:      llvm.func internal @__tvm_ffi_init
// CHECK-LLVM:      llvm.call @TVMFFIFunctionGetGlobal
// CHECK-LLVM:      llvm.call @TVMFFIFunctionGetGlobal
// CHECK-LLVM:      llvm.func internal @__tvm_ffi_fini
// CHECK-LLVM:      llvm.call @TVMFFIObjectDecRef
// CHECK-LLVM:      llvm.call @TVMFFIObjectDecRef
// CHECK-LLVM:      llvm.mlir.global_ctors
// CHECK-LLVM:      llvm.mlir.global_dtors

// CHECK-TVM-FFI-LABEL: func.func @multiple_names
func.func @multiple_names(%arg0: !tvm_ffi.any, %arg1: !tvm_ffi.any) -> !tvm_ffi.any {
  // CHECK-TVM-FFI:      [[ADDRB:%.*]] = llvm.mlir.addressof @__tvm_ffi_beta
  // CHECK-TVM-FFI:      [[ADDRA:%.*]] = llvm.mlir.addressof @__tvm_ffi_alpha
  // CHECK-TVM-FFI:      [[HA:%.*]] = tvm_ffi.load [[ADDRA]]
  // CHECK-TVM-FFI-NEXT: tvm_ffi.function_call [[HA]](
  // CHECK-TVM-FFI:      [[HB:%.*]] = tvm_ffi.load [[ADDRB]]
  // CHECK-TVM-FFI-NEXT: tvm_ffi.function_call [[HB]](
  // CHECK-TVM-FFI:      [[HA2:%.*]] = tvm_ffi.load [[ADDRA]]
  // CHECK-TVM-FFI-NEXT: tvm_ffi.function_call [[HA2]](
  // CHECK-LLVM:      llvm.call @TVMFFIFunctionCall
  // CHECK-LLVM:      llvm.call @TVMFFIFunctionCall
  // CHECK-LLVM:      llvm.call @TVMFFIFunctionCall
  %0 = tvm_ffi.call "alpha"(%arg0) : (!tvm_ffi.any) -> !tvm_ffi.any
  %1 = tvm_ffi.call "beta"(%arg1) : (!tvm_ffi.any) -> !tvm_ffi.any
  %2 = tvm_ffi.call "alpha"(%arg0, %arg1) : (!tvm_ffi.any, !tvm_ffi.any) -> !tvm_ffi.any
  return %2 : !tvm_ffi.any
}

// -----
// CHECK-TVM-FFI:      llvm.mlir.global internal @__tvm_ffi_single

// CHECK-TVM-FFI:      llvm.func {{.*}} @__tvm_ffi_init
// CHECK-TVM-FFI:         [[H:%.*]] = tvm_ffi.function_get_global "single"
// CHECK-TVM-FFI:         llvm.return

// CHECK-TVM-FFI:      llvm.func {{.*}} @__tvm_ffi_fini
// CHECK-TVM-FFI:         [[HL:%.*]] = tvm_ffi.load %{{.*}} : !llvm.ptr -> !tvm_ffi.object_handle
// CHECK-TVM-FFI:         tvm_ffi.object_dec_ref [[HL]]
// CHECK-TVM-FFI:         llvm.return

// CHECK-LLVM:      llvm.func @TVMFFIFunctionCall
// CHECK-LLVM:      llvm.func @TVMFFIObjectDecRef
// CHECK-LLVM:      llvm.mlir.global internal constant @__libtriton_tvm_ffi_func_name_single
// CHECK-LLVM:      llvm.func @TVMFFIFunctionGetGlobal
// CHECK-LLVM:      llvm.mlir.global internal @__tvm_ffi_single

// CHECK-LLVM:      llvm.func {{.*}} @__tvm_ffi_init
// CHECK-LLVM:         llvm.mlir.addressof
// CHECK-LLVM:         llvm.call @TVMFFIFunctionGetGlobal

// CHECK-LLVM:      llvm.func {{.*}} @__tvm_ffi_fini
// CHECK-LLVM:         llvm.load
// CHECK-LLVM:         llvm.call @TVMFFIObjectDecRef
// CHECK-LLVM:         llvm.return

// CHECK-LLVM:      llvm.mlir.global_ctors
// CHECK-LLVM:      llvm.mlir.global_dtors

// CHECK-TVM-FFI-LABEL: func.func @single_arg
func.func @single_arg(%arg0: !tvm_ffi.any) -> !tvm_ffi.any {
  // CHECK-TVM-FFI:      [[ADDR:%.*]] = llvm.mlir.addressof @__tvm_ffi_single
  // CHECK-TVM-FFI:      [[HANDLE:%.*]] = tvm_ffi.load [[ADDR]]
  // CHECK-TVM-FFI:      tvm_ffi.function_call [[HANDLE]](
  // CHECK-LLVM:      llvm.call @TVMFFIFunctionCall
  %0 = tvm_ffi.call "single"(%arg0) : (!tvm_ffi.any) -> !tvm_ffi.any
  return %0 : !tvm_ffi.any
}
