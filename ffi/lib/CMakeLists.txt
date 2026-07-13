# ===----------------------------------------------------------------------===//
#
# Part of the Trident project, under the MIT License. SPDX-License-Identifier:
# MIT
#
# ===----------------------------------------------------------------------===//

add_library(TridentFFI SHARED Exception.cc)
target_include_directories(TridentFFI PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
  TridentFFI
  PUBLIC tvm_ffi::header
  PRIVATE tvm_ffi::shared)

install(
  TARGETS TridentFFI
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
