#include "libtriton-core/Runtime/Runtime.h"
#include "c10/cuda/CUDAFunctions.h"
#include "c10/cuda/CUDAStream.h"

LIBTRITON_CORE_RUNTIME_EXPORT DLDevice __libtriton_get_current_device() {
  return {kDLCUDA, c10::cuda::current_device()};
}

LIBTRITON_CORE_RUNTIME_EXPORT cudaStream_t
__libtriton_get_current_stream(c10::DeviceIndex device_index) {
  return c10::cuda::getCurrentCUDAStream(device_index).stream();
}
