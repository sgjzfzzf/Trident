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
  // CHECK: %[[S0:.*]] = torch_ext.get_current_stream : !llvm.ptr
  // CHECK: torch_ext.triton_kernel_launch @kernel<%[[S0]] : !llvm.ptr> blocks in(%{{.*}}, %{{.*}}, %{{.*}}) threads in(%{{.*}}, %{{.*}}, %{{.*}}) : index

  // -----

  func.func @async_kernel_launch_for_gpu_launch_func(%gx: index, %gy: index, %gz: index,
                                                    %bx: index, %by: index, %bz: index) {
    gpu.launch_func @kernels::@kernel blocks in (%gx, %gy, %gz) threads in (%bx, %by, %bz)
    return
  }

  // CHECK-LABEL: func.func @async_kernel_launch_for_gpu_launch_func
  // CHECK: %[[S1:.*]] = torch_ext.get_current_stream : !llvm.ptr
  // CHECK: gpu.launch_func <%[[S1]] : !llvm.ptr> @kernels::@kernel blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}})

  // -----

  func.func @keep_existing_async_stream(%gx: index, %gy: index, %gz: index,
                                  %bx: index, %by: index, %bz: index) {
    %stream = torch_ext.get_current_stream : !llvm.ptr
    torch_ext.triton_kernel_launch @kernel <%stream : !llvm.ptr> blocks in(%gx, %gy, %gz) threads in(%bx, %by, %bz) : index
    gpu.launch_func < %stream : !llvm.ptr > @kernels::@kernel blocks in (%gx, %gy, %gz) threads in (%bx, %by, %bz)
    return
  }

  // CHECK-LABEL: func.func @keep_existing_async_stream
  // CHECK: %[[S2:.*]] = torch_ext.get_current_stream : !llvm.ptr
  // CHECK-NOT: torch_ext.get_current_stream : !llvm.ptr
  // CHECK: torch_ext.triton_kernel_launch @kernel<%[[S2]] : !llvm.ptr> blocks in(%{{.*}}, %{{.*}}, %{{.*}}) threads in(%{{.*}}, %{{.*}}, %{{.*}}) : index
  // CHECK: gpu.launch_func <%[[S2]] : !llvm.ptr> @kernels::@kernel blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}})
}