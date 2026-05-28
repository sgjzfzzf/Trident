// RUN: libtriton-core-opt -torchext-async-kernel-launch %s | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @kernel() kernel {
      gpu.return
    }
  }

  func.func @async_kernel_launch_for_torch_ext(%gx: index, %gy: index, %gz: index,
                                               %bx: index, %by: index, %bz: index) {
    torch_ext.triton_kernel_launch @kernel blocks in(%gx, %gy, %gz) threads in(%bx, %by, %bz) : index
    return
  }

  // CHECK-LABEL: func.func @async_kernel_launch_for_torch_ext
  // CHECK: %[[S0:.*]] = torch_ext.get_current_stream device = -1 : !gpu.async.token
  // CHECK: {{.*}} = torch_ext.triton_kernel_launch async[%[[S0]] : !gpu.async.token] @kernel blocks in(%{{.*}}, %{{.*}}, %{{.*}}) threads in(%{{.*}}, %{{.*}}, %{{.*}}) -> !gpu.async.token : index

  // -----

  func.func @async_kernel_launch_for_gpu_launch_func(%gx: index, %gy: index, %gz: index,
                                                    %bx: index, %by: index, %bz: index) {
    gpu.launch_func @kernels::@kernel blocks in (%gx, %gy, %gz) threads in (%bx, %by, %bz)
    return
  }

  // CHECK-LABEL: func.func @async_kernel_launch_for_gpu_launch_func
  // CHECK: %[[T0:.*]] = torch_ext.get_current_stream device = -1 : !gpu.async.token
  // CHECK: %[[T1:.*]] = gpu.launch_func async [%[[T0]]] @kernels::@kernel blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}})

  // -----

  func.func @keep_existing_async_stream(%gx: index, %gy: index, %gz: index,
                                  %bx: index, %by: index, %bz: index) {
    %stream = torch_ext.get_current_stream device = -1 : !gpu.async.token
    %triton_result = torch_ext.triton_kernel_launch async[%stream : !gpu.async.token] @kernel blocks in(%gx, %gy, %gz) threads in(%bx, %by, %bz) -> !gpu.async.token : index
    %gpu_result = gpu.launch_func async [%stream] @kernels::@kernel blocks in (%gx, %gy, %gz) threads in (%bx, %by, %bz)
    return
  }

  // CHECK-LABEL: func.func @keep_existing_async_stream
  // CHECK: %[[S2:.*]] = torch_ext.get_current_stream device = -1 : !gpu.async.token
  // CHECK: torch_ext.triton_kernel_launch async[%[[S2]] : !gpu.async.token] @kernel blocks in(%{{.*}}, %{{.*}}, %{{.*}}) threads in(%{{.*}}, %{{.*}}, %{{.*}}) -> !gpu.async.token : index
  // CHECK: gpu.launch_func async
  // CHECK-SAME: @kernels::@kernel blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}})
}