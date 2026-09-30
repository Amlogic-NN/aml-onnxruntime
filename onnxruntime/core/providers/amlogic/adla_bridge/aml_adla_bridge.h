#pragma once

#include <string>

namespace amlogic::adla_bridge {

// Compile the Amlogic reference-style ops_json directly, without producing or
// reparsing a TFLite file.
// Returns false and fills error on parse, mapping, or compiler failures.
bool CompileOpsJsonToAdla(const std::string& ops_json,
                          const std::string& output_dir,
                          const std::string& model_name,
                          const std::string& target,
                          std::string* adla_path,
                          std::string* error);

}  // namespace amlogic::adla_bridge
