#pragma once

#include <cstdlib>
#include <cstring>
#include <iostream>

namespace onnxruntime {
namespace amlnn_logging {
inline int Level() {
  static const int level = [] {
    const char* value = std::getenv("ORT_AMLNN_LOG_LEVEL");
    if (value == nullptr || *value == '\0') return 0;
    if (std::strcmp(value, "error") == 0 || std::strcmp(value, "ERROR") == 0) return 0;
    if (std::strcmp(value, "info") == 0 || std::strcmp(value, "INFO") == 0) return 1;
    if (std::strcmp(value, "debug") == 0 || std::strcmp(value, "DEBUG") == 0) return 2;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end != value && *end == '\0') return parsed < 0 ? 0 : parsed > 2 ? 2
                                                                         : static_cast<int>(parsed);
    return 0;
  }();
  return level;
}
inline bool Enabled(int level) { return level == 0 || Level() >= level; }
}  // namespace amlnn_logging
}  // namespace onnxruntime

#define LOGE                                     \
  if (!::onnxruntime::amlnn_logging::Enabled(0)) \
    ;                                            \
  else                                           \
    std::cerr << "[AMLNN][E] "
#define LOGI                                     \
  if (!::onnxruntime::amlnn_logging::Enabled(1)) \
    ;                                            \
  else                                           \
    std::cerr << "[AMLNN][I] "
#define LOGD                                     \
  if (!::onnxruntime::amlnn_logging::Enabled(2)) \
    ;                                            \
  else                                           \
    std::cerr << "[AMLNN][D] "
