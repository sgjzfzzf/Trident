# --- Generate aten.gen.cc -------------------------------------------------
set(ATEN_GEN_CC ${CMAKE_CURRENT_BINARY_DIR}/aten.gen.cc)
set(ATENGEN_PY ${CMAKE_CURRENT_SOURCE_DIR}/python/atengen.py)

add_custom_command(
  OUTPUT ${ATEN_GEN_CC}
  COMMAND ${Python3_EXECUTABLE} ${ATENGEN_PY} --out ${ATEN_GEN_CC}
  COMMAND clang-format -i ${ATEN_GEN_CC}
  DEPENDS ${ATENGEN_PY} ${CMAKE_CURRENT_SOURCE_DIR}/python/templates/aten.cc.j2
  COMMENT "Generating ${ATEN_GEN_CC}")

# Static library containing the generated wrappers and Value<> conversion logic.
add_library(AtenGen STATIC ${ATEN_GEN_CC})

target_include_directories(AtenGen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(AtenGen PUBLIC ${TORCH_LIBRARIES} tvm_ffi::header)

# The aoti_torch headers are part of libtorch's include path, which is already
# set via TORCH_INCLUDE_DIRS at the project level.  No extra include_directories
# needed here.
