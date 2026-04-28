// RUN: libtriton-core-opt -torchext-normalize-operands %s | FileCheck %s

// Case 1: function arguments are builtin tensor; launch argument is first
// converted to vtensor via torch_c.from_builtin_tensor. The pass should rewrite
// launch args back to ordinary tensor values.
func.func @normalize_launch_from_builtin_operand(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %n: i32) {
      %vt0 = torch_c.from_builtin_tensor %arg0 : tensor<4xf32> -> !torch.vtensor<[4], f32>
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : i32
      torch_ext.torch_kernel_launch @kernel blocks in(%c1, %c1, %c1) threads in(%c1, %c1, %c1) dynamic_shared_memory_size %c0 args(%vt0, %arg1, %n : !torch.vtensor<[4], f32>, tensor<4xf32>, i32) : index
      return
}

// CHECK-LABEL: func.func @normalize_launch_from_builtin_operand
// CHECK: torch_ext.torch_kernel_launch @kernel blocks in(%{{.*}}, %{{.*}},
// %{{.*}}) threads in(%{{.*}}, %{{.*}}, %{{.*}}) dynamic_shared_memory_size
// %{{.*}} args(%arg0, %arg1, %n : tensor<4xf32>, tensor<4xf32>, i32) : index

// -----

// Case 2: operand comes from to_builtin_tensor -> from_builtin_tensor chain.
// The pass should still strip the from_builtin_tensor and keep builtin tensor.
func.func @normalize_launch_from_to_chain(%arg0: !torch.vtensor<[4], f32>, %n: i32) {
      %t0 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[4], f32> -> tensor<4xf32>
      %vt0 = torch_c.from_builtin_tensor %t0 : tensor<4xf32> -> !torch.vtensor<[4], f32>
      %c1 = arith.constant 1 : index
      torch_ext.torch_kernel_launch @kernel blocks in(%c1, %c1, %c1) threads in(%c1, %c1, %c1) args(%vt0, %n : !torch.vtensor<[4], f32>, i32) : index
      return
}

// CHECK-LABEL: func.func @normalize_launch_from_to_chain
// CHECK: %[[T0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[4],f32> -> tensor<4xf32>
// CHECK: torch_ext.torch_kernel_launch @kernel blocks in(%{{.*}}, %{{.*}}, %{{.*}}) threads in(%{{.*}}, %{{.*}}, %{{.*}}) args(%[[T0]], %{{.*}} : tensor<4xf32>, i32) : index

// -----

// Case 3: non-from_builtin_tensor operands should be preserved.
func.func @preserve_non_from_builtin_operand(%arg0: !torch.vtensor<[4], f32>, %n: i32) {
      %c1 = arith.constant 1 : index
      torch_ext.torch_kernel_launch @kernel blocks in(%c1, %c1, %c1) threads in(%c1, %c1, %c1) args(%arg0, %n : !torch.vtensor<[4], f32>, i32) : index
      return
}

// CHECK-LABEL: func.func @preserve_non_from_builtin_operand
// CHECK: torch_ext.torch_kernel_launch @kernel blocks in(%{{.*}}, %{{.*}}, %{{.*}}) threads in(%{{.*}}, %{{.*}}, %{{.*}}) args(%arg0, %{{.*}} : !torch.vtensor<[4],f32>, i32) : index
