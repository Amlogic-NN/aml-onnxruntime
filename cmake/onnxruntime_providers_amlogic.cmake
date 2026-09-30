# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  if (onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_EXTENDED_MINIMAL_BUILD)
    message(FATAL_ERROR "Amlogic NPU EP can not be used in a basic minimal build. Please build with '--minimal_build extended'")
  endif()

  add_compile_definitions(USE_AMLOGIC=1)

  file(GLOB_RECURSE onnxruntime_providers_amlogic_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/amlogic/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/amlogic/*.cc"
  )
  list(FILTER onnxruntime_providers_amlogic_srcs EXCLUDE REGEX "/adla_bridge/")

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_amlogic_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_amlogic ${onnxruntime_providers_amlogic_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_amlogic
    onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers
    Boost::mp11 safeint_interface nlohmann_json::nlohmann_json
  )
  add_dependencies(onnxruntime_providers_amlogic onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_include_directories(onnxruntime_providers_amlogic PRIVATE ${ONNXRUNTIME_ROOT})
  set_target_properties(onnxruntime_providers_amlogic PROPERTIES FOLDER "ONNXRuntime" LINKER_LANGUAGE CXX)

  # The optional TFLite refactor implementation is not embedded in this
  # build. Keep retained debug source compatible with ORT's -Werror policy.
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(onnxruntime_providers_amlogic PRIVATE
      -Wno-error=unused-const-variable)
  endif()

  if (onnxruntime_USE_AMLNN AND ANDROID)
    if (NOT AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR)
      set(AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR "$ENV{AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR}")
    endif()
    if (NOT AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR)
      message(FATAL_ERROR "AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR is required when the Android Amlogic converter is enabled")
    endif()
    add_library(aml_adla_bridge STATIC
      "${ONNXRUNTIME_ROOT}/core/providers/amlogic/adla_bridge/aml_adla_bridge.cc"
      "${ONNXRUNTIME_ROOT}/core/providers/amlogic/adla_bridge/aml_adla_bridge_c.cc"
      "${ONNXRUNTIME_ROOT}/core/providers/amlogic/adla_bridge/ops_json_to_aml.cc"
      "${ONNXRUNTIME_ROOT}/core/providers/amlogic/compiler_core_support/context_binary_info.cc"
      "${ONNXRUNTIME_ROOT}/core/providers/amlogic/compiler_core_support/adla_log.cc")
    target_include_directories(aml_adla_bridge
      PUBLIC
        "${ONNXRUNTIME_ROOT}/core/providers/amlogic/adla_bridge"
        "${ONNXRUNTIME_ROOT}/core/providers/amlogic/compiler_core_support"
      PRIVATE
        "${AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR}")
    target_link_libraries(aml_adla_bridge PUBLIC nlohmann_json::nlohmann_json)
    if (CMAKE_DL_LIBS)
      target_link_libraries(aml_adla_bridge PUBLIC ${CMAKE_DL_LIBS})
    endif()
    if (ANDROID)
      target_link_libraries(aml_adla_bridge PUBLIC log)
    endif()
    set_target_properties(aml_adla_bridge PROPERTIES
      FOLDER "ONNXRuntime"
      LINKER_LANGUAGE CXX)
  endif()

  if(NOT onnxruntime_AMLOGIC_NPU_STUB)
    target_compile_definitions(onnxruntime_providers_amlogic PRIVATE ORT_AMLOGIC_USE_VENDOR_SDK=1)
    if(onnxruntime_AMLOGIC_NPU_SDK_INCLUDE_DIR)
      target_include_directories(onnxruntime_providers_amlogic PRIVATE ${onnxruntime_AMLOGIC_NPU_SDK_INCLUDE_DIR})
    endif()
    if(onnxruntime_AMLOGIC_NPU_LIBRARY)
      target_link_libraries(onnxruntime_providers_amlogic PRIVATE ${onnxruntime_AMLOGIC_NPU_LIBRARY})
    else()
      message(WARNING "onnxruntime_AMLOGIC_NPU_STUB=OFF but onnxruntime_AMLOGIC_NPU_LIBRARY is not set. Add the vendor SDK library after wiring amlogic_backend.cc.")
    endif()
  endif()

  if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_providers_amlogic EXPORT ${PROJECT_NAME}Targets
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
