// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is intentionally included from amlogic_qlinear_model_info.cc
// inside namespace reference_style_internal. It groups the shared helpers used
// by the reference_style compat/collector/postprocess modules.

std::string GetWrappedOpType(const ordered_json& op_json) {
  if (!op_json.is_object() || op_json.empty()) {
    return {};
  }
  return op_json.begin().key();
}

ordered_json* GetWrappedPayload(ordered_json& op_json) {
  if (!op_json.is_object() || op_json.empty()) {
    return nullptr;
  }
  return &op_json.begin().value();
}

const ordered_json* GetWrappedPayload(const ordered_json& op_json) {
  if (!op_json.is_object() || op_json.empty()) {
    return nullptr;
  }
  return &op_json.begin().value();
}

bool IsQuantNoneValue(const ordered_json& value) {
  return value.is_string() && value.get<std::string>() == "None";
}

bool HasRealQuantInfo(const ordered_json& tensor_info) {
  return tensor_info.contains("quant_scale") &&
         tensor_info.contains("zero_point") &&
         !IsQuantNoneValue(tensor_info["quant_scale"]);
}

std::vector<std::string> TensorInfoKeysWithPrefix(const ordered_json& payload,
                                                  std::string_view prefix) {
  std::vector<std::pair<int, std::string>> indexed_keys;
  for (auto it = payload.begin(); it != payload.end(); ++it) {
    const std::string& key = it.key();
    if (key == prefix) {
      indexed_keys.emplace_back(0, key);
      continue;
    }
    if (StartsWith(key, prefix) && key.size() > prefix.size() &&
        key[prefix.size()] == '_') {
      const std::string suffix = key.substr(prefix.size() + 1);
      int index = 0;
      if (!suffix.empty()) {
        index = std::atoi(suffix.c_str());
      }
      indexed_keys.emplace_back(index, key);
    }
  }

  std::sort(indexed_keys.begin(), indexed_keys.end(),
            [](const auto& lhs, const auto& rhs) {
              if (lhs.first != rhs.first) {
                return lhs.first < rhs.first;
              }
              return lhs.second < rhs.second;
            });

  std::vector<std::string> keys;
  keys.reserve(indexed_keys.size());
  for (const auto& indexed_key : indexed_keys) {
    keys.push_back(indexed_key.second);
  }
  return keys;
}

std::vector<std::string> InputTensorInfoKeys(const ordered_json& payload) {
  return TensorInfoKeysWithPrefix(payload, "input_tensor_info");
}

std::vector<std::string> OutputTensorInfoKeys(const ordered_json& payload) {
  return TensorInfoKeysWithPrefix(payload, "output_tensor_info");
}

const ordered_json* FindNamedTensorInfo(const ordered_json& payload,
                                        const std::vector<std::string>& keys,
                                        const std::string& tensor_name) {
  for (const auto& key : keys) {
    auto it = payload.find(key);
    if (it != payload.end() &&
        it->is_object() &&
        it->contains("tensor_name") &&
        (*it)["tensor_name"] == tensor_name) {
      return &(*it);
    }
  }
  return nullptr;
}

ordered_json* FindNamedTensorInfo(ordered_json& payload,
                                  const std::vector<std::string>& keys,
                                  const std::string& tensor_name) {
  for (const auto& key : keys) {
    auto it = payload.find(key);
    if (it != payload.end() &&
        it->is_object() &&
        it->contains("tensor_name") &&
        (*it)["tensor_name"] == tensor_name) {
      return &(*it);
    }
  }
  return nullptr;
}

void CopyQuantInfo(const ordered_json& from, ordered_json& to) {
  if (from.contains("quant_scale")) {
    to["quant_scale"] = from["quant_scale"];
  }
  if (from.contains("zero_point")) {
    to["zero_point"] = from["zero_point"];
  }
}

void CopyTensorType(const ordered_json& from, ordered_json& to) {
  if (from.contains("tensor_type")) {
    to["tensor_type"] = from["tensor_type"];
  }
}

int FindProducerIndex(const ordered_json& ops_json,
                      const std::string& tensor_name,
                      int end_exclusive) {
  const int end = std::min(end_exclusive, static_cast<int>(ops_json.size()));
  for (int i = end - 1; i >= 0; --i) {
    const auto* payload = GetWrappedPayload(ops_json[static_cast<size_t>(i)]);
    if (payload == nullptr) {
      continue;
    }
    const auto keys = OutputTensorInfoKeys(*payload);
    if (FindNamedTensorInfo(*payload, keys, tensor_name) != nullptr) {
      return i;
    }
  }
  return -1;
}

const ordered_json* FindProducerTensorInfo(const ordered_json& ops_json,
                                           const std::string& tensor_name,
                                           int end_exclusive) {
  const int producer_index = FindProducerIndex(ops_json, tensor_name, end_exclusive);
  if (producer_index < 0) {
    return nullptr;
  }

  const auto* payload = GetWrappedPayload(ops_json[static_cast<size_t>(producer_index)]);
  if (payload == nullptr) {
    return nullptr;
  }

  const auto keys = OutputTensorInfoKeys(*payload);
  return FindNamedTensorInfo(*payload, keys, tensor_name);
}

int FindFirstConsumerIndex(const ordered_json& ops_json,
                           const std::string& tensor_name,
                           int start_inclusive) {
  const int start = std::max(0, start_inclusive);
  for (int i = start; i < static_cast<int>(ops_json.size()); ++i) {
    const auto* payload = GetWrappedPayload(ops_json[static_cast<size_t>(i)]);
    if (payload == nullptr) {
      continue;
    }
    const auto keys = InputTensorInfoKeys(*payload);
    if (FindNamedTensorInfo(*payload, keys, tensor_name) != nullptr) {
      return i;
    }
  }
  return -1;
}

Status MatchQuantScale(ordered_json& ops_json,
                       const std::string& tensor_direction,
                       size_t op_index,
                       const std::string& tensor_key) {
  auto* payload = GetWrappedPayload(ops_json[op_index]);
  ORT_RETURN_IF_NOT(payload != nullptr, "Malformed reference-style op payload.");
  auto tensor_it = payload->find(tensor_key);
  ORT_RETURN_IF_NOT(tensor_it != payload->end() && tensor_it->is_object(),
                    "Missing tensor info key in reference-style op: ", tensor_key);

  ordered_json& tensor_info = *tensor_it;
  ORT_RETURN_IF_NOT(tensor_info.contains("tensor_name"),
                    "Missing tensor name in reference-style tensor info.");
  const std::string tensor_name = tensor_info["tensor_name"].get<std::string>();

  if (tensor_direction == "input") {
    for (int i = static_cast<int>(op_index) - 1; i >= 0; --i) {
      const auto* candidate_payload = GetWrappedPayload(ops_json[static_cast<size_t>(i)]);
      if (candidate_payload == nullptr) {
        continue;
      }
      const auto output_keys = OutputTensorInfoKeys(*candidate_payload);
      const auto* producer_tensor = FindNamedTensorInfo(*candidate_payload, output_keys, tensor_name);
      if (producer_tensor != nullptr && HasRealQuantInfo(*producer_tensor)) {
        CopyQuantInfo(*producer_tensor, tensor_info);
        break;
      }
    }
  } else if (tensor_direction == "output") {
    for (size_t i = op_index + 1; i < ops_json.size(); ++i) {
      const auto* candidate_payload = GetWrappedPayload(ops_json[i]);
      if (candidate_payload == nullptr) {
        continue;
      }
      const auto input_keys = InputTensorInfoKeys(*candidate_payload);
      const auto* consumer_tensor = FindNamedTensorInfo(*candidate_payload, input_keys, tensor_name);
      if (consumer_tensor != nullptr && HasRealQuantInfo(*consumer_tensor)) {
        CopyQuantInfo(*consumer_tensor, tensor_info);
        break;
      }
    }
  }

  return Status::OK();
}

std::vector<int> JsonShapeToVector(const ordered_json& tensor_info) {
  std::vector<int> shape;
  const auto shape_it = tensor_info.find("tensor_shape");
  if (shape_it == tensor_info.end() || !shape_it->is_array()) {
    return shape;
  }

  shape.reserve(shape_it->size());
  for (const auto& dim : *shape_it) {
    if (dim.is_number_integer()) {
      shape.push_back(dim.get<int>());
    }
  }
  return shape;
}
