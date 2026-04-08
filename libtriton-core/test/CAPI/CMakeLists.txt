add_executable(libtriton-core-capi-dlpack-test dlpack.c)
target_link_libraries(libtriton-core-capi-dlpack-test PRIVATE
  LibTritonCoreCAPI
  MLIRCAPIIR
)

add_executable(libtriton-core-capi-tvmffi-test tvmffi.c)
target_link_libraries(libtriton-core-capi-tvmffi-test PRIVATE
  LibTritonCoreCAPI
  MLIRCAPIIR
)

set(CAPI_TEST_TARGETS
  libtriton-core-capi-dlpack-test
  libtriton-core-capi-tvmffi-test
)

set(CAPI_TEST_TARGETS ${CAPI_TEST_TARGETS} PARENT_SCOPE)
