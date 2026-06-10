#ifndef LIBTRITON_CORE_DIALECT_TORCHEXT_IR_TORCHEXTBASE_TD_
#define LIBTRITON_CORE_DIALECT_TORCHEXT_IR_TORCHEXTBASE_TD_

include "mlir/IR/DialectBase.td"

def TorchExt_Dialect : Dialect {
  let name = "torchext";
  let cppNamespace = "::libtriton::torchext";
  let summary = "TorchExt dialect";
  // clang-format off
  let description = [{
    The TorchExt dialect models the stable C API defined in
    torch/csrc/inductor/aoti_torch/c/shim.h. This C shim layer provides ABI
    compatibility for AOTInductor-compiled model.so, which must NOT directly
    depend on aten/c10 C++ APIs (which have no ABI guarantees). The dialect
    covers the full shim surface including:
      - Device, dtype, layout, and memory format constant getters
      - Tensor introspection (data ptr, sizes, strides, dtype, device, layout)
      - Tensor construction (empty_strided, from_blob, reinterpret_tensor, clone)
      - Tensor manipulation (copy_, as_strided, assign, zero_, scatter)
      - Scalar-to-tensor and tensor-to-scalar conversion
      - c10::IValue construction and destruction
      - Record function guards (aoti_record_function_start/end)
      - Device/stream guards (create_device_guard, get_current_stream)
      - CUDA-specific ops (caching allocator, stream guards)
      - Proxy executor dispatch
    All ops in this dialect map to their aoti_torch_* counterpart and share
    the same error handling via AOTITorchError return codes.
  }];
  // clang-format on
}

#endif // LIBTRITON_CORE_DIALECT_TORCHEXT_IR_TORCHEXTBASE_TD_
