// Minimal CPU tensor add e2e: tensor_add_kernel(x, y) returns x + y.

func.func @tensor_add_impl(%x: memref<4xi64>, %y: memref<4xi64>) -> memref<4xi64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %out = memref.alloc() : memref<4xi64>

  %x0 = memref.load %x[%c0] : memref<4xi64>
  %y0 = memref.load %y[%c0] : memref<4xi64>
  %sum0 = arith.addi %x0, %y0 : i64
  memref.store %sum0, %out[%c0] : memref<4xi64>

  %x1 = memref.load %x[%c1] : memref<4xi64>
  %y1 = memref.load %y[%c1] : memref<4xi64>
  %sum1 = arith.addi %x1, %y1 : i64
  memref.store %sum1, %out[%c1] : memref<4xi64>

  %x2 = memref.load %x[%c2] : memref<4xi64>
  %y2 = memref.load %y[%c2] : memref<4xi64>
  %sum2 = arith.addi %x2, %y2 : i64
  memref.store %sum2, %out[%c2] : memref<4xi64>

  %x3 = memref.load %x[%c3] : memref<4xi64>
  %y3 = memref.load %y[%c3] : memref<4xi64>
  %sum3 = arith.addi %x3, %y3 : i64
  memref.store %sum3, %out[%c3] : memref<4xi64>

  return %out : memref<4xi64>
}

func.func @tensor_add_kernel(%x: memref<4xi64>, %y: memref<4xi64>) -> memref<4xi64>
  attributes {
    tvm_ffi.emit_tvm_ffi_interface
  }
{
  %tmp = call @tensor_add_impl(%x, %y) : (memref<4xi64>, memref<4xi64>) -> memref<4xi64>
  return %tmp : memref<4xi64>
}
