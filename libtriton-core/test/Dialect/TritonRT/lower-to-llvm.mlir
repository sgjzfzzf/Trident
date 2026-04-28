// RUN: libtriton-core-opt %s -convert-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @kernel(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
        kernel {
      gpu.return
    }
  }
  gpu.module @buf_kernels {
    gpu.func @buf_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr)
        kernel {
      gpu.return
    }
  }

  // CHECK-LABEL: llvm.func @lower_scalar_operand
  func.func @lower_scalar_operand(
      %gx: i64, %gy: i64, %gz: i64, %bx: i64, %by: i64, %bz: i64,
      %arg0: i32) {
    torch_ext.triton_kernel_launch @kernels::@kernel
        blocks in(%gx, %gy, %gz) threads in(%bx, %by, %bz)
        args(%arg0 : i32) : i64
    return
    // CHECK-NOT: builtin.unrealized_conversion_cast
    // CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: gpu.launch_func @kernels::@kernel
    // CHECK-SAME: args(%{{.*}} : i32, %[[NULL]] : !llvm.ptr, %[[NULL]] : !llvm.ptr)
  }

  // memref<?xf32> is flattened to (ptr, ptr, i64, i64, i64) by convert-func-to-llvm.
  // The descriptor is reconstructed via insertvalue, and field [1] (aligned ptr)
  // is extracted and passed as the kernel argument.
  // CHECK-LABEL: llvm.func @lower_memref_operand
  func.func @lower_memref_operand(
      %gx: i64, %gy: i64, %gz: i64, %bx: i64, %by: i64, %bz: i64,
      %buf: memref<?xf32>) {
    torch_ext.triton_kernel_launch @buf_kernels::@buf_kernel
        blocks in(%gx, %gy, %gz) threads in(%bx, %by, %bz)
        args(%buf : memref<?xf32>) : i64
    return
    // CHECK-NOT: builtin.unrealized_conversion_cast
    // The descriptor is rebuilt from the flattened args, then field [1] extracted.
    // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[1]
    // CHECK: %[[APTR:.*]] = llvm.extractvalue %{{.*}}[1]
    // CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK: gpu.launch_func @buf_kernels::@buf_kernel
    // CHECK-SAME: args(%[[APTR]] : !llvm.ptr, %[[NULL]] : !llvm.ptr, %[[NULL]] : !llvm.ptr)
  }
}
