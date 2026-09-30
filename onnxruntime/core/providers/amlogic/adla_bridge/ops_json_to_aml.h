#pragma once

#include <string>

#include "context_binary_info.h"

namespace amlogic::adla_bridge {

bool BuildAmlModelFromOpsJson(const std::string& ops_json, const std::string& target,
                              AML_Model* model, std::string* error);

}  // namespace amlogic::adla_bridge

