# --- Generate aten.gen.cc -------------------------------------------------
set(ATENGEN_CC ${CMAKE_CURRENT_BINARY_DIR}/aten.gen.cc)
set(ATENGEN_PY ${CMAKE_CURRENT_SOURCE_DIR}/python/atengen.py)

add_custom_command(
  OUTPUT ${ATENGEN_CC}
  COMMAND ${Python3_EXECUTABLE} ${ATENGEN_PY} --out ${ATENGEN_CC}
  COMMAND clang-format -i ${ATENGEN_CC}
  DEPENDS ${ATENGEN_PY} ${CMAKE_CURRENT_SOURCE_DIR}/templates/aten.cc.j2
  COMMENT "Generating ${ATENGEN_CC}")

add_library(TridentCoreRuntime SHARED Exception.cc Runtime.cc ${ATENGEN_CC})
target_include_directories(TridentCoreRuntime
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
  TridentCoreRuntime
  PUBLIC ${TORCH_LIBRARIES} CUDA::cudart CUDA::cuda_driver tvm_ffi::header
  PRIVATE tvm_ffi::shared)

install(
  TARGETS TridentCoreRuntime
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
