# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_EXTENDED_MINIMAL_BUILD)
  message(FATAL_ERROR "Amlogic EP requires an extended minimal build or a full build")
endif()

if(NOT AMLNN_NNSDK2_INCLUDE_DIR)
  set(AMLNN_NNSDK2_INCLUDE_DIR "$ENV{AMLNN_NNSDK2_INCLUDE_DIR}")
endif()
if(NOT AMLNN_NNSDK2_INCLUDE_DIR)
  message(FATAL_ERROR "AMLNN_NNSDK2_INCLUDE_DIR is required when onnxruntime_USE_AMLOGIC is ON")
endif()
if(NOT EXISTS "${AMLNN_NNSDK2_INCLUDE_DIR}/nnsdk2.h")
  message(FATAL_ERROR "Missing NNSDK2 header: ${AMLNN_NNSDK2_INCLUDE_DIR}/nnsdk2.h")
endif()

# A single provider contains the working AMLNN EP/runtime and Linghua exporter.
# Keep the bridge separate as an internal static target, never a shared SO.
file(GLOB_RECURSE onnxruntime_providers_amlogic_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/amlogic/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/amlogic/*.cc")
list(FILTER onnxruntime_providers_amlogic_srcs EXCLUDE REGEX "/adla_bridge/")
source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_amlogic_srcs})
onnxruntime_add_static_library(onnxruntime_providers_amlogic ${onnxruntime_providers_amlogic_srcs})
onnxruntime_add_include_to_target(onnxruntime_providers_amlogic
  onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB}
  flatbuffers::flatbuffers Boost::mp11 safeint_interface nlohmann_json::nlohmann_json)
add_dependencies(onnxruntime_providers_amlogic onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnxruntime_providers_amlogic PRIVATE
  ${ONNXRUNTIME_ROOT} ${AMLNN_NNSDK2_INCLUDE_DIR})
target_compile_definitions(onnxruntime_providers_amlogic PRIVATE USE_AMLOGIC=1)
set_target_properties(onnxruntime_providers_amlogic PROPERTIES
  FOLDER "ONNXRuntime" LINKER_LANGUAGE CXX)
if(CMAKE_DL_LIBS)
  target_link_libraries(onnxruntime_providers_amlogic PRIVATE ${CMAKE_DL_LIBS})
endif()

# Retained exporter debug code must remain compatible with ORT's -Werror policy.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(onnxruntime_providers_amlogic PRIVATE -Wno-error=unused-const-variable)
endif()

# Preserve the existing platform boundary: Android supports ONNX conversion;
# other runtime builds use AMLNN_MODEL_PATH to load a precompiled ADLA.
if(ANDROID)
  if(NOT AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR)
    set(AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR "$ENV{AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR}")
  endif()
  foreach(_core_header IN ITEMS compiler_core_api.h compiler_core_context.h)
    if(NOT EXISTS "${AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR}/${_core_header}")
      message(FATAL_ERROR "Missing Core SDK header: ${AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR}/${_core_header}")
    endif()
  endforeach()
  add_library(aml_adla_bridge STATIC
    "${ONNXRUNTIME_ROOT}/core/providers/amlogic/adla_bridge/aml_adla_bridge.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/amlogic/adla_bridge/aml_adla_bridge_c.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/amlogic/adla_bridge/ops_json_to_aml.cc")
  target_include_directories(aml_adla_bridge
    PUBLIC "${ONNXRUNTIME_ROOT}/core/providers/amlogic/adla_bridge"
    PRIVATE "${AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR}")
  target_link_libraries(aml_adla_bridge PUBLIC nlohmann_json::nlohmann_json log)
  if(CMAKE_DL_LIBS)
    target_link_libraries(aml_adla_bridge PUBLIC ${CMAKE_DL_LIBS})
  endif()
  set_target_properties(aml_adla_bridge PROPERTIES
    FOLDER "ONNXRuntime" POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE CXX)
  target_compile_definitions(onnxruntime_providers_amlogic PRIVATE ORT_AMLOGIC_USE_CONVERTER=1)
  target_link_libraries(onnxruntime_providers_amlogic PRIVATE aml_adla_bridge)
endif()

if(NOT onnxruntime_BUILD_SHARED_LIB)
  install(TARGETS onnxruntime_providers_amlogic EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(FILES
  "${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/amlogic/amlogic_provider_factory.h"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/)
