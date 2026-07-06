add_library(TridentCoreRuntime SHARED Runtime.cc)
target_link_libraries(
  TridentCoreRuntime
  PUBLIC -Wl,--whole-archive
         AtenGen
         -Wl,--no-whole-archive
         CUDA::cudart
         CUDA::cuda_driver
         LLVMSupport
         tvm_ffi::header)

install(
  TARGETS TridentCoreRuntime
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
