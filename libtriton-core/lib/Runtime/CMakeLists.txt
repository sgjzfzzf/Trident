add_library(LibTritonCoreRuntime SHARED Runtime.cc)
target_link_libraries(
  LibTritonCoreRuntime PUBLIC CUDA::cudart CUDA::cuda_driver LLVMSupport
                              tvm_ffi::header)

install(
  TARGETS LibTritonCoreRuntime
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
