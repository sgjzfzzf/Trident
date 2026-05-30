// RUN: libtriton-core-opt %s --libtriton-one-shot-bufferize='bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map' | FileCheck %s --check-prefix=IDENTITY
// RUN: libtriton-core-opt %s --libtriton-one-shot-bufferize='bufferize-function-boundaries=1 function-boundary-type-conversion=infer-layout-map' | FileCheck %s --check-prefix=INFER

module {
  // Static + dynamic tensor arguments/results should bufferize at function
  // boundaries.
  func.func @mixed_static_dynamic_identity(%arg0: tensor<4xf32>, %arg1: tensor<?xf32>) -> (tensor<4xf32>, tensor<?xf32>) {
    return %arg0, %arg1 : tensor<4xf32>, tensor<?xf32>
  }

  // Call graph rewriting should keep signatures and call operands/results
  // consistent after tensor->memref conversion.
  func.func @call_mixed_static_dynamic_identity(%arg0: tensor<4xf32>, %arg1: tensor<?xf32>) -> (tensor<4xf32>, tensor<?xf32>) {
    %0:2 = call @mixed_static_dynamic_identity(%arg0, %arg1) : (tensor<4xf32>, tensor<?xf32>) -> (tensor<4xf32>, tensor<?xf32>)
    return %0#0, %0#1 : tensor<4xf32>, tensor<?xf32>
  }

  // Single-result dynamic tensor return should also be converted.
  func.func @dynamic_identity(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    return %arg0 : tensor<?xf32>
  }

  // Tensor update that keeps both updated tensor and original tensor live.
  // Bufferization should allocate a new buffer and copy from original.
  func.func @insert_conflict(%arg0: tensor<4xf32>, %v: f32) -> (tensor<4xf32>, tensor<4xf32>) {
    %c0 = arith.constant 0 : index
    %m = tensor.insert %v into %arg0[%c0] : tensor<4xf32>
    return %m, %arg0 : tensor<4xf32>, tensor<4xf32>
  }

  // Construct a brand new dynamic tensor from tensor.empty + tensor.insert.
  // Bufferization should allocate with dynamic sizes.
  func.func @build_from_empty(%n: index, %v: f32) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %t0 = tensor.empty(%n) : tensor<?xf32>
    %t1 = tensor.insert %v into %t0[%c0] : tensor<?xf32>
    return %t1 : tensor<?xf32>
  }
}

// IDENTITY-LABEL: func.func @mixed_static_dynamic_identity(
// IDENTITY-SAME: %[[S0:.+]]: memref<4xf32>, %[[D0:.+]]: memref<?xf32>) -> (memref<4xf32>, memref<?xf32>)
// IDENTITY: return %[[S0]], %[[D0]] : memref<4xf32>, memref<?xf32>

// IDENTITY-LABEL: func.func @call_mixed_static_dynamic_identity(
// IDENTITY-SAME: %[[S1:.+]]: memref<4xf32>, %[[D1:.+]]: memref<?xf32>) -> (memref<4xf32>, memref<?xf32>)
// IDENTITY: %[[R:.+]]:2 = call @mixed_static_dynamic_identity(%[[S1]], %[[D1]]) : (memref<4xf32>, memref<?xf32>) -> (memref<4xf32>, memref<?xf32>)
// IDENTITY: return %[[R]]#0, %[[R]]#1 : memref<4xf32>, memref<?xf32>

// IDENTITY-LABEL: func.func @dynamic_identity(
// IDENTITY-SAME: %[[D2:.+]]: memref<?xf32>) -> memref<?xf32>
// IDENTITY: return %[[D2]] : memref<?xf32>

// IDENTITY-LABEL: func.func @insert_conflict(
// IDENTITY-SAME: %[[SRC0:.+]]: memref<4xf32>, %[[V0:.+]]: f32) -> (memref<4xf32>, memref<4xf32>)
// IDENTITY: %[[A0:.+]] = memref.alloc() {{.*}} : memref<4xf32>
// IDENTITY: memref.copy %[[SRC0]], %[[A0]] : memref<4xf32> to memref<4xf32>
// IDENTITY: memref.store %[[V0]], %[[A0]]
// IDENTITY: return %[[A0]], %[[SRC0]] : memref<4xf32>, memref<4xf32>

// IDENTITY-LABEL: func.func @build_from_empty(
// IDENTITY-SAME: %[[N0:.+]]: index, %[[V1:.+]]: f32) -> memref<?xf32>
// IDENTITY: %[[A1:.+]] = memref.alloc(%[[N0]]) {{.*}} : memref<?xf32>
// IDENTITY: memref.store %[[V1]], %[[A1]]
// IDENTITY: return %[[A1]] : memref<?xf32>

// INFER-LABEL: func.func @mixed_static_dynamic_identity(
// INFER-SAME: %[[S3:.+]]: memref<4xf32, strided<[?], offset: ?>>, %[[D3:.+]]: memref<?xf32, strided<[?], offset: ?>>) -> (memref<4xf32, strided<[?], offset: ?>>, memref<?xf32, strided<[?], offset: ?>>)
// INFER: return %[[S3]], %[[D3]] : memref<4xf32, strided<[?], offset: ?>>, memref<?xf32, strided<[?], offset: ?>>

// INFER-LABEL: func.func @call_mixed_static_dynamic_identity(
// INFER-SAME: %[[S4:.+]]: memref<4xf32, strided<[?], offset: ?>>, %[[D4:.+]]: memref<?xf32, strided<[?], offset: ?>>) -> (memref<4xf32, strided<[?], offset: ?>>, memref<?xf32, strided<[?], offset: ?>>)
// INFER: %[[R2:.+]]:2 = call @mixed_static_dynamic_identity(%[[S4]], %[[D4]]) : (memref<4xf32, strided<[?], offset: ?>>, memref<?xf32, strided<[?], offset: ?>>) -> (memref<4xf32, strided<[?], offset: ?>>, memref<?xf32, strided<[?], offset: ?>>)
// INFER: return %[[R2]]#0, %[[R2]]#1 : memref<4xf32, strided<[?], offset: ?>>, memref<?xf32, strided<[?], offset: ?>>

// INFER-LABEL: func.func @dynamic_identity(
// INFER-SAME: %[[D5:.+]]: memref<?xf32, strided<[?], offset: ?>>) -> memref<?xf32, strided<[?], offset: ?>>
// INFER: return %[[D5]] : memref<?xf32, strided<[?], offset: ?>>

// INFER-LABEL: func.func @insert_conflict(
// INFER-SAME: %[[SRC1:.+]]: memref<4xf32, strided<[?], offset: ?>>, %[[V2:.+]]: f32)
// INFER: %[[A2:.+]] = memref.alloc() {{.*}} : memref<4xf32>
// INFER: memref.copy %[[SRC1]], %[[A2]] : memref<4xf32, strided<[?], offset: ?>> to memref<4xf32>
// INFER: memref.store %[[V2]], %[[A2]]

// INFER-LABEL: func.func @build_from_empty(
// INFER-SAME: %[[N1:.+]]: index, %[[V3:.+]]: f32) -> memref<?xf32>
// INFER: %[[A3:.+]] = memref.alloc(%[[N1]]) {{.*}} : memref<?xf32>
// INFER: memref.store %[[V3]], %[[A3]]
// INFER: return %[[A3]] : memref<?xf32>
