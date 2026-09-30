if (NOT AMLNN_NNSDK2_INCLUDE_DIR)
  set(AMLNN_NNSDK2_INCLUDE_DIR "$ENV{AMLNN_NNSDK2_INCLUDE_DIR}")
endif()
if (NOT AMLNN_NNSDK2_INCLUDE_DIR)
  message(FATAL_ERROR "AMLNN_NNSDK2_INCLUDE_DIR is required when onnxruntime_USE_AMLNN is ON")
endif()
file(GLOB_RECURSE onnxruntime_providers_amlnn_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/amlnn/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/amlnn/*.cc")
onnxruntime_add_static_library(onnxruntime_providers_amlnn ${onnxruntime_providers_amlnn_srcs})
onnxruntime_add_include_to_target(onnxruntime_providers_amlnn
  onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB}
  flatbuffers::flatbuffers Boost::mp11 nlohmann_json::nlohmann_json)
target_include_directories(onnxruntime_providers_amlnn PRIVATE ${ONNXRUNTIME_ROOT} ${AMLNN_NNSDK2_INCLUDE_DIR})
if (onnxruntime_USE_AMLOGIC AND ANDROID)
  target_include_directories(onnxruntime_providers_amlnn PRIVATE
    "${ONNXRUNTIME_ROOT}/core/providers/amlogic/adla_bridge"
    "${ONNXRUNTIME_ROOT}/core/providers/amlogic/compiler_core_support"
    "${AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR}")
  target_compile_definitions(onnxruntime_providers_amlnn PRIVATE ORT_AMLNN_USE_AMLOGIC_CONVERTER=1)
endif()
set_target_properties(onnxruntime_providers_amlnn PROPERTIES FOLDER "ONNXRuntime" LINKER_LANGUAGE CXX)

if (NOT onnxruntime_BUILD_SHARED_LIB)
  install(TARGETS onnxruntime_providers_amlnn EXPORT ${PROJECT_NAME}Targets
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

install(FILES ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/amlnn/amlnn_provider_factory.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/)

if (CMAKE_DL_LIBS)
  target_link_libraries(onnxruntime_providers_amlnn PRIVATE ${CMAKE_DL_LIBS})
endif()
if (onnxruntime_USE_AMLOGIC AND ANDROID)
  target_link_libraries(onnxruntime_providers_amlnn PRIVATE aml_adla_bridge)
endif()
