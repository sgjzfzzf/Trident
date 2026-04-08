// RUN: libtriton-core-opt %s -emit-tvm-ffi-interface | FileCheck %s --check-prefix=CHECK-TVM-FFI
// RUN: libtriton-core-opt %s -emit-tvm-ffi-interface -convert-to-llvm -reconcile-unrealized-casts | mlir-opt -finalize-memref-to-llvm -reconcile-unrealized-casts | FileCheck %s --check-prefix=CHECK-LLVM
// CHECK-TVM-FFI-NOT: func.func @__tvm_ffi_plain
// CHECK-TVM-FFI-LABEL: func.func @id_i64
func.func @id_i64(%x: i64) -> i64 attributes {tvm_ffi.emit_tvm_ffi_interface} {
  return %x : i64
}

func.func @plain(%x: i64) -> i64 {
  return %x : i64
}

// CHECK-TVM-FFI-LABEL: func.func @pick_float
func.func @pick_float(%i: i64, %f: f64, %p: !llvm.ptr) -> f64 attributes {tvm_ffi.emit_tvm_ffi_interface} {
  return %f : f64
}

// CHECK-TVM-FFI-LABEL: func.func @ret_tmp_memref
func.func @ret_tmp_memref() -> memref<4xf32> attributes {tvm_ffi.emit_tvm_ffi_interface} {
  %tmp = memref.alloca() : memref<4xf32>
  return %tmp : memref<4xf32>
}

// CHECK-TVM-FFI-LABEL: func.func @__tvm_ffi_id_i64(
// CHECK-TVM-FFI-SAME: %[[HANDLE:.*]]: !llvm.ptr, %[[ARGS:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RESULT:.*]]: !llvm.ptr) -> i32 {
// CHECK-TVM-FFI: %[[ARG0_INDEX:.*]] = arith.constant 0 : i64
// CHECK-TVM-FFI: %[[ARG0_PTR:.*]] = llvm.getelementptr %[[ARGS]][%[[ARG0_INDEX]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-TVM-FFI: %[[ARG0_ANY_LLVM:.*]] = llvm.load %[[ARG0_PTR]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-TVM-FFI: %[[ARG0_ANY:.*]] = builtin.unrealized_conversion_cast %[[ARG0_ANY_LLVM]] : !llvm.struct<(i32, i32, i64)> to !tvm_ffi.any
// CHECK-TVM-FFI: %[[ARG0:.*]] = tvm_ffi.to_int %[[ARG0_ANY]] : !tvm_ffi.any -> i64
// CHECK-TVM-FFI: %[[CALLEE_RET:.*]] = call @id_i64(%[[ARG0]]) : (i64) -> i64
// CHECK-TVM-FFI: %[[BOXED_RET:.*]] = tvm_ffi.from_int %[[CALLEE_RET]] : i64 -> !tvm_ffi.any
// CHECK-TVM-FFI: %[[RET_PTR:.*]] = llvm.getelementptr %[[RESULT]][{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-TVM-FFI: %[[BOXED_RET_LLVM:.*]] = builtin.unrealized_conversion_cast %[[BOXED_RET]] : !tvm_ffi.any to !llvm.struct<(i32, i32, i64)>
// CHECK-TVM-FFI: llvm.store %[[BOXED_RET_LLVM]], %[[RET_PTR]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK-TVM-FFI: %[[SUCCESS:.*]] = arith.constant 0 : i32
// CHECK-TVM-FFI: return %[[SUCCESS]] : i32

// CHECK-TVM-FFI-LABEL: func.func @__tvm_ffi_pick_float(
// CHECK-TVM-FFI-SAME: %[[MHANDLE:.*]]: !llvm.ptr, %[[MARGS:.*]]: !llvm.ptr, %[[MNUM_ARGS:.*]]: i32, %[[MRESULT:.*]]: !llvm.ptr) -> i32 {
// CHECK-TVM-FFI: %[[MARG0_INDEX:.*]] = arith.constant 0 : i64
// CHECK-TVM-FFI: %[[MARG0_PTR:.*]] = llvm.getelementptr %[[MARGS]][%[[MARG0_INDEX]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-TVM-FFI: %[[MARG0_ANY_LLVM:.*]] = llvm.load %[[MARG0_PTR]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-TVM-FFI: %[[MARG0_ANY:.*]] = builtin.unrealized_conversion_cast %[[MARG0_ANY_LLVM]] : !llvm.struct<(i32, i32, i64)> to !tvm_ffi.any
// CHECK-TVM-FFI: %[[MARG0:.*]] = tvm_ffi.to_int %[[MARG0_ANY]] : !tvm_ffi.any -> i64
// CHECK-TVM-FFI: %[[MARG1_INDEX:.*]] = arith.constant 1 : i64
// CHECK-TVM-FFI: %[[MARG1_PTR:.*]] = llvm.getelementptr %[[MARGS]][%[[MARG1_INDEX]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-TVM-FFI: %[[MARG1_ANY_LLVM:.*]] = llvm.load %[[MARG1_PTR]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-TVM-FFI: %[[MARG1_ANY:.*]] = builtin.unrealized_conversion_cast %[[MARG1_ANY_LLVM]] : !llvm.struct<(i32, i32, i64)> to !tvm_ffi.any
// CHECK-TVM-FFI: %[[MARG1:.*]] = tvm_ffi.to_float %[[MARG1_ANY]] : !tvm_ffi.any -> f64
// CHECK-TVM-FFI: %[[MARG2_INDEX:.*]] = arith.constant 2 : i64
// CHECK-TVM-FFI: %[[MARG2_PTR:.*]] = llvm.getelementptr %[[MARGS]][%[[MARG2_INDEX]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-TVM-FFI: %[[MARG2_ANY_LLVM:.*]] = llvm.load %[[MARG2_PTR]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-TVM-FFI: %[[MARG2_ANY:.*]] = builtin.unrealized_conversion_cast %[[MARG2_ANY_LLVM]] : !llvm.struct<(i32, i32, i64)> to !tvm_ffi.any
// CHECK-TVM-FFI: %[[MARG2:.*]] = tvm_ffi.to_str %[[MARG2_ANY]] : !tvm_ffi.any -> !llvm.ptr
// CHECK-TVM-FFI: %[[MCALLEE_RET:.*]] = call @pick_float(%[[MARG0]], %[[MARG1]], %[[MARG2]]) : (i64, f64, !llvm.ptr) -> f64
// CHECK-TVM-FFI: %[[MBOXED_RET:.*]] = tvm_ffi.from_float %[[MCALLEE_RET]] : f64 -> !tvm_ffi.any
// CHECK-TVM-FFI: %[[MRET_PTR:.*]] = llvm.getelementptr %[[MRESULT]][{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-TVM-FFI: %[[MBOXED_RET_LLVM:.*]] = builtin.unrealized_conversion_cast %[[MBOXED_RET]] : !tvm_ffi.any to !llvm.struct<(i32, i32, i64)>
// CHECK-TVM-FFI: llvm.store %[[MBOXED_RET_LLVM]], %[[MRET_PTR]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK-TVM-FFI: %[[MSUCCESS:.*]] = arith.constant 0 : i32
// CHECK-TVM-FFI: return %[[MSUCCESS]] : i32

// CHECK-TVM-FFI-LABEL: func.func @__tvm_ffi_ret_tmp_memref(
// CHECK-TVM-FFI-SAME: %[[RHANDLE:.*]]: !llvm.ptr, %[[RARGS:.*]]: !llvm.ptr, %[[RNUM_ARGS:.*]]: i32, %[[RRESULT:.*]]: !llvm.ptr) -> i32 {
// CHECK-TVM-FFI: %[[RTMP:.*]] = call @ret_tmp_memref() : () -> memref<4xf32>
// CHECK-TVM-FFI: %[[ROWNED:.*]] = dlpack.from_memref_owned %[[RTMP]] : memref<4xf32> -> !dlpack.managed_tensor
// CHECK-TVM-FFI: %[[RZERO:.*]] = arith.constant 0 : i32
// CHECK-TVM-FFI: %[[RH:.*]] = tvm_ffi.tensor_from_dlpack %[[ROWNED]], %[[RZERO]], %[[RZERO]] : !dlpack.managed_tensor, i32, i32 -> !tvm_ffi.object_handle
// CHECK-TVM-FFI: %[[RBOXED:.*]] = tvm_ffi.from_tensor %[[RH]] : !tvm_ffi.object_handle -> !tvm_ffi.any
// CHECK-TVM-FFI: %[[RRET_PTR:.*]] = llvm.getelementptr %[[RRESULT]][{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-TVM-FFI: %[[RBOXED_LLVM:.*]] = builtin.unrealized_conversion_cast %[[RBOXED]] : !tvm_ffi.any to !llvm.struct<(i32, i32, i64)>
// CHECK-TVM-FFI: llvm.store %[[RBOXED_LLVM]], %[[RRET_PTR]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK-TVM-FFI: %[[RSUCCESS:.*]] = arith.constant 0 : i32
// CHECK-TVM-FFI: return %[[RSUCCESS]] : i32

// CHECK-LLVM-NOT: func.func @__tvm_ffi_plain
// CHECK-LLVM-LABEL: func.func @id_i64(
// CHECK-LLVM-SAME: %{{.*}}: i64) -> i64 {
// CHECK-LLVM-NOT: tvm_ffi.emit_tvm_ffi_interface
// CHECK-LLVM-NOT: builtin.unrealized_conversion_cast
// CHECK-LLVM: return

// CHECK-LLVM-LABEL: func.func @pick_float(
// CHECK-LLVM-SAME: %{{.*}}: i64, %{{.*}}: f64, %{{.*}}: !llvm.ptr) -> f64 {
// CHECK-LLVM-NOT: tvm_ffi.emit_tvm_ffi_interface
// CHECK-LLVM-NOT: builtin.unrealized_conversion_cast
// CHECK-LLVM: return

// CHECK-LLVM-LABEL: func.func @ret_tmp_memref() -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> {
// CHECK-LLVM: llvm.alloca {{.*}} x f32 : (i64) -> !llvm.ptr
// CHECK-LLVM: llvm.insertvalue {{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-LLVM: llvm.insertvalue {{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-LLVM: return {{.*}} : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>

// CHECK-LLVM-LABEL: func.func @__tvm_ffi_id_i64(
// CHECK-LLVM-SAME: %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i32, %{{.*}}: !llvm.ptr) -> i32 {
// CHECK-LLVM-NOT: builtin.unrealized_conversion_cast
// CHECK-LLVM: llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-LLVM: llvm.load {{.*}} : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-LLVM: llvm.extractvalue {{.*}}[2] : !llvm.struct<(i32, i32, i64)>
// CHECK-LLVM: call @id_i64({{.*}}) : (i64) -> i64
// CHECK-LLVM: llvm.mlir.poison : !llvm.struct<(i32, i32, i64)>
// CHECK-LLVM: llvm.insertvalue {{.*}}[2] : !llvm.struct<(i32, i32, i64)>
// CHECK-LLVM: llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-LLVM: llvm.store {{.*}} : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK-LLVM: return {{.*}} : i32

// CHECK-LLVM-LABEL: func.func @__tvm_ffi_pick_float(
// CHECK-LLVM-SAME: %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i32, %{{.*}}: !llvm.ptr) -> i32 {
// CHECK-LLVM-NOT: builtin.unrealized_conversion_cast
// CHECK-LLVM: llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-LLVM: llvm.load {{.*}} : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-LLVM: llvm.extractvalue {{.*}}[2] : !llvm.struct<(i32, i32, i64)>
// CHECK-LLVM: llvm.bitcast {{.*}} : i64 to f64
// CHECK-LLVM: llvm.inttoptr {{.*}} : i64 to !llvm.ptr
// CHECK-LLVM: call @pick_float({{.*}}) : (i64, f64, !llvm.ptr) -> f64
// CHECK-LLVM: llvm.bitcast {{.*}} : f64 to i64
// CHECK-LLVM: llvm.mlir.poison : !llvm.struct<(i32, i32, i64)>
// CHECK-LLVM: llvm.insertvalue {{.*}}[2] : !llvm.struct<(i32, i32, i64)>
// CHECK-LLVM: llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-LLVM: llvm.store {{.*}} : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK-LLVM: return {{.*}} : i32

// CHECK-LLVM-LABEL: func.func @__tvm_ffi_ret_tmp_memref(
// CHECK-LLVM-SAME: %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i32, %{{.*}}: !llvm.ptr) -> i32 {
// CHECK-LLVM: call @ret_tmp_memref() : () -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-LLVM: llvm.call @malloc
// CHECK-LLVM: llvm.call @malloc
// CHECK-LLVM: llvm.call @malloc
// CHECK-LLVM: llvm.store {{.*}} : !llvm.struct<(struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>, ptr, ptr)>, !llvm.ptr
// CHECK-LLVM: llvm.call @TVMFFITensorFromDLPack
// CHECK-LLVM: llvm.ptrtoint {{.*}} : !llvm.ptr to i64
// CHECK-LLVM: llvm.store {{.*}} : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK-LLVM: return {{.*}} : i32
