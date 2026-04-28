// RUN: libtriton-core-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @triton_kernel_launch_with_tensor_and_vtensor
func.func @triton_kernel_launch_with_tensor_and_vtensor(%gx: index, %gy: index, %gz: index, %bx: index, %by: index, %bz: index, %smem: i32, %arg0: tensor<4xf32>, %arg1: !torch.vtensor<[4], f32>) {
  // CHECK: torch_ext.triton_kernel_launch @kernel_tensor_vtensor blocks
  // in(%{{.*}}, %{{.*}}, %{{.*}}) threads in(%{{.*}}, %{{.*}}, %{{.*}})
  // dynamic_shared_memory_size %{{.*}} args(%{{.*}}, %{{.*}} : tensor<4xf32>,
  // !torch.vtensor<[4],f32>) : index
    torch_ext.triton_kernel_launch @kernel_tensor_vtensor blocks in(%gx, %gy, %gz) threads in(%bx, %by, %bz) dynamic_shared_memory_size %smem args(%arg0, %arg1 : tensor<4xf32>, !torch.vtensor<[4], f32>) : index
    return
}

// -----

// CHECK-LABEL: func.func @triton_kernel_launch_without_dynamic_smem
func.func @triton_kernel_launch_without_dynamic_smem(%gx: i64, %gy: i64, %gz: i64, %bx: i64, %by: i64, %bz: i64, %arg0: i32) {
  // CHECK: torch_ext.triton_kernel_launch @kernel_no_smem blocks in(%{{.*}},
  // %{{.*}}, %{{.*}}) threads in(%{{.*}}, %{{.*}}, %{{.*}}) args(%{{.*}} : i32)
  // : i64
    torch_ext.triton_kernel_launch @kernel_no_smem blocks in(%gx, %gy, %gz) threads in(%bx, %by, %bz) args(%arg0 : i32) : i64
    return
}

// -----

// CHECK-LABEL: func.func @triton_kernel_launch_without_kernel_args
func.func @triton_kernel_launch_without_kernel_args(%gx: i32, %gy: i32, %gz: i32, %bx: i32, %by: i32, %bz: i32) {
  // CHECK: torch_ext.triton_kernel_launch @kernel_no_args blocks in(%{{.*}},
  // %{{.*}}, %{{.*}}) threads in(%{{.*}}, %{{.*}}, %{{.*}}) : i32
    torch_ext.triton_kernel_launch @kernel_no_args blocks in(%gx, %gy, %gz) threads in(%bx, %by, %bz) : i32
    return
}
