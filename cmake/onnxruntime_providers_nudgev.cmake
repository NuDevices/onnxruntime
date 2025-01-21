# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

add_definitions(-DUSE_NUDGEV=1)

# Gather all source files
file(GLOB_RECURSE
  onnxruntime_providers_nudgev_cc_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/nudgev/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/nudgev/*.cc"
)

if (onnxruntime_BUILD_UNIT_TESTS)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Disable the uninitialized warning for tests
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/test/providers/cpu/ml/write_scores_test.cc
      PROPERTIES
      COMPILE_FLAGS "-Wno-error=maybe-uninitialized"
    )
  endif()
endif()

source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_nudgev_cc_srcs})

# Create the provider library
onnxruntime_add_static_library(onnxruntime_providers_nudgev ${onnxruntime_providers_nudgev_cc_srcs})

# Add include dependencies
onnxruntime_add_include_to_target(onnxruntime_providers_nudgev
  onnxruntime_common
  onnxruntime_framework
  onnx
  onnx_proto
  ${PROTOBUF_LIB}
  flatbuffers::flatbuffers
)

# Add dependencies
add_dependencies(onnxruntime_providers_nudgev onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})

# Set properties
set_target_properties(onnxruntime_providers_nudgev PROPERTIES FOLDER "ONNXRuntime")
target_include_directories(onnxruntime_providers_nudgev PRIVATE ${ONNXRUNTIME_ROOT})
target_include_directories(onnxruntime_providers_nudgev PRIVATE
  ${ONNXRUNTIME_ROOT}
  ${eigen_SOURCE_DIR}
)
set_target_properties(onnxruntime_providers_nudgev PROPERTIES LINKER_LANGUAGE CXX)

# Installation rules if building static library
if (NOT onnxruntime_BUILD_SHARED_LIB)
  install(TARGETS onnxruntime_providers_nudgev
          ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
