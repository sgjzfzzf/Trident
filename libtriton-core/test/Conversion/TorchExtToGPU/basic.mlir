// RUN: libtriton-core-opt %s --convert-torchext-to-gpu | FileCheck %s

// CHECK-LABEL: module attributes {gpu.container_module}
// CHECK:      llvm.func @aoti_torch_get_current_stream(i32, !llvm.ptr) -> i32
// CHECK:      llvm.func @aoti_torch_get_current_device_index(!llvm.ptr) -> i32
// CHECK:      gpu.module @kernel {
// CHECK:        gpu.func @entry(%{{.*}}: !llvm.ptr, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64)
// CHECK-SAME:   kernel attributes {gpu.binary = ""}
// CHECK-NEXT:     gpu.return
// CHECK-NEXT:   }
// CHECK:      func.func @test_kernel_launch
// CHECK-SAME: %[[TENS_ARG:.*]]: !torch.vtensor<[4],f32>
// CHECK-SAME: %[[SCAL_ARG:.*]]: !torch.int
// CHECK:      %[[SCALAR_CAST:.*]] = builtin.unrealized_conversion_cast %[[SCAL_ARG]] : !torch.int to i64
// CHECK:      %[[TENSOR_PTR:.*]] = builtin.unrealized_conversion_cast %[[TENS_ARG]] : !torch.vtensor<[4],f32> to !llvm.ptr
// CHECK:      %[[C32:.*]] = arith.constant 32 : index
// CHECK:      %[[C16:.*]] = arith.constant 16 : index
// CHECK:      %[[C128:.*]] = arith.constant 128 : index
// CHECK:      %[[C1:.*]] = arith.constant 1 : index
// CHECK:      %[[SHMEM:.*]] = arith.constant 16384 : i32
// CHECK:      %[[DLTENSOR_PTR:.*]] = llvm.getelementptr %[[TENSOR_PTR]]{{\[}}24] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK:      %[[DATA_GEP:.*]] = llvm.getelementptr %[[DLTENSOR_PTR]]{{\[}}0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
// CHECK:      %[[DATA_PTR:.*]] = llvm.load %[[DATA_GEP]] : !llvm.ptr -> !llvm.ptr
// CHECK:      %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:      %[[DEV_IDX_SLOT:.*]] = llvm.alloca %[[ONE]] x i32 : (i64) -> !llvm.ptr
// CHECK:      llvm.call @aoti_torch_get_current_device_index(%[[DEV_IDX_SLOT]]) : (!llvm.ptr) -> i32
// CHECK:      %[[DEVICE_IDX:.*]] = llvm.load %[[DEV_IDX_SLOT]] : !llvm.ptr -> i32
// CHECK:      %[[STRM_SLOT:.*]] = llvm.alloca %{{.*}} x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK:      llvm.call @aoti_torch_get_current_stream(%[[DEVICE_IDX]], %[[STRM_SLOT]]) : (i32, !llvm.ptr) -> i32
// CHECK:      %[[ASYNC_OBJ:.*]] = llvm.load %[[STRM_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK:      gpu.launch_func <%[[ASYNC_OBJ]] : !llvm.ptr> @kernel::@entry
// CHECK:      blocks in (%[[C32]], %[[C16]], %[[C1]])
// CHECK:      threads in (%[[C128]], %[[C1]], %[[C1]])
// CHECK:      dynamic_shared_memory_size %[[SHMEM]]
// CHECK:      args(%[[DATA_PTR]] : !llvm.ptr, %[[SCALAR_CAST]] : i64, {{%.*}} : i64, {{%.*}} : i64)

// CHECK-NOT: torchext.triton_kernel_launch
module attributes { gpu.container_module } {
  gpu.module @kernel {
    gpu.func @entry(%arg0: !llvm.ptr, %arg1: i64, %arg2: i64, %arg3: i64)
        attributes { gpu.binary = "", gpu.kernel } {
      gpu.return
    }
  }

  func.func @test_kernel_launch(
    %tensor: !torch.vtensor<[4],f32>,
    %scalar: !torch.int) {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %shmem = arith.constant 16384 : i32

    torchext.triton_kernel_launch @kernel::@entry
      blocks in (%c32, %c16, %c1) threads in (%c128, %c1, %c1)
      dynamic_shared_memory_size %shmem
      args (%tensor, %scalar : !torch.vtensor<[4],f32>, !torch.int)
      : index
    func.return
  }
}
