#pragma once

#include <string>

#include "bridge_ir.h"

namespace amlnn::adla_bridge {

bool BuildAmlModelFromOpsJson(const std::string& ops_json, const std::string& target,
                              AML_Model* model, std::string* error);

}  // namespace amlnn::adla_bridge

