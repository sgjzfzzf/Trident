add_mlir_public_c_api_library(
  TridentCoreCAPI
  Dialects.cc
  Registration.cc
  TVMFFITypes.cc
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ENABLE_AGGREGATION
  LINK_LIBS
  PUBLIC
  DLPackDialect
  MLIRIR
  MLIRSupport
  TVMFFIDialect
  TridentUtilsRegistration)

install(
  TARGETS TridentCoreCAPI
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
