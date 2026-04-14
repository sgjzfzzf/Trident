find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(_libtriton_core_python_package_dir
  ${CMAKE_BINARY_DIR}/python_packages/libtriton-core
)

set(_libtriton_python_source_dir
  ${CMAKE_SOURCE_DIR}/../python
)

add_test(
  NAME libtriton-core-python-tvm-ffi-add-numpy
  COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH=${_libtriton_python_source_dir}:${_libtriton_core_python_package_dir}
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tvm_ffi_add_numpy_e2e.py
)

add_test(
  NAME libtriton-core-python-tvm-ffi-add-torch-cpu
  COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH=${_libtriton_python_source_dir}:${_libtriton_core_python_package_dir}
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tvm_ffi_add_torch_cpu_e2e.py
)

add_test(
  NAME libtriton-core-python-torch-dialect
  COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH=${_libtriton_python_source_dir}:${_libtriton_core_python_package_dir}
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/torch_dialect_pipeline.py
)

set_tests_properties(libtriton-core-python-tvm-ffi-add-numpy PROPERTIES
  ENVIRONMENT "PYTHONPATH=${_libtriton_python_source_dir}:${_libtriton_core_python_package_dir}"
)

set_tests_properties(libtriton-core-python-tvm-ffi-add-torch-cpu PROPERTIES
  ENVIRONMENT "PYTHONPATH=${_libtriton_python_source_dir}:${_libtriton_core_python_package_dir}"
)

set_tests_properties(libtriton-core-python-torch-dialect PROPERTIES
  ENVIRONMENT "PYTHONPATH=${_libtriton_python_source_dir}:${_libtriton_core_python_package_dir}"
)
