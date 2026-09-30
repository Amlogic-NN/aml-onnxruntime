#include "aml_adla_bridge_c.h"

#include <cstring>
#include <string>

#include "aml_adla_bridge.h"

namespace {

void WriteString(const std::string& value, char* destination, unsigned long capacity) {
  if (destination == nullptr || capacity == 0) {
    return;
  }
  const size_t count = value.size() < capacity - 1 ? value.size() : capacity - 1;
  std::memcpy(destination, value.data(), count);
  destination[count] = '\0';
}

}  // namespace

extern "C" int aml_compile_ops_json_to_adla(
    const char* ops_json,
    const char* output_dir,
    const char* model_name,
    const char* target,
    char* adla_path,
    unsigned long adla_path_capacity,
    char* error_message,
    unsigned long error_message_capacity) {
  if (adla_path != nullptr && adla_path_capacity != 0) {
    adla_path[0] = '\0';
  }
  if (error_message != nullptr && error_message_capacity != 0) {
    error_message[0] = '\0';
  }

  if (ops_json == nullptr || output_dir == nullptr || model_name == nullptr || target == nullptr) {
    WriteString("null input argument", error_message, error_message_capacity);
    return 1;
  }

  std::string result_path;
  std::string error;
  const bool ok = amlogic::adla_bridge::CompileOpsJsonToAdla(
      ops_json, output_dir, model_name, target, &result_path, &error);
  if (!ok) {
    WriteString(error.empty() ? "ADLA compilation failed" : error,
                error_message, error_message_capacity);
    return 1;
  }

  if (adla_path == nullptr || adla_path_capacity <= result_path.size()) {
    WriteString("adla_path buffer is too small", error_message, error_message_capacity);
    return 1;
  }
  WriteString(result_path, adla_path, adla_path_capacity);
  return 0;
}
