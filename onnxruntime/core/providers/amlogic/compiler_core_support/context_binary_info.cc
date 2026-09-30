/*******************************************************************************
 * Copyright (C) 2023 Amlogic, Inc. All rights reserved.
 ******************************************************************************/

#include "context_binary_info.h"

#include "adla_log.h"

#include <sstream>
#include <iostream>
#include <string>

// ================== 序列化 ==================
std::string SerializeModel(const AML_Model &model)
{
    std::ostringstream oss(std::ios::binary);

    auto write_string = [&](const std::string &s)
    {
        size_t len = s.size();
        oss.write((char *)&len, sizeof(len));
        oss.write(s.data(), len);
    };

    auto write_tensor = [&](const AML_Tensor &t)
    {
        write_string(t.name);

        size_t dims_size = t.dims.size();
        oss.write((char *)&dims_size, sizeof(dims_size));
        oss.write((char *)t.dims.data(), dims_size * sizeof(int));

        oss.write((char *)&t.dtype, sizeof(t.dtype));
        oss.write((char *)&t.is_const, sizeof(t.is_const));
        oss.write((char *)&t.tensor_index, sizeof(t.tensor_index));

        // === 写入 zp ===
        size_t zp_size = t.zp.size();
        oss.write((char *)&zp_size, sizeof(zp_size));
        if (zp_size > 0)
            oss.write((char *)t.zp.data(), zp_size * sizeof(int));

        // === 写入 scale ===
        size_t scale_size = t.scale.size();
        oss.write((char *)&scale_size, sizeof(scale_size));
        if (scale_size > 0)
            oss.write((char *)t.scale.data(), scale_size * sizeof(float));

        size_t data_size = t.data.size();
        oss.write((char *)&data_size, sizeof(data_size));
        if (data_size > 0)
            oss.write((char *)t.data.data(), data_size);

        oss.write((char *)&t.size, sizeof(t.size));
    };

    auto write_node = [&](const AML_Node &n)
    {
        oss.write((char *)&n.Node_type, sizeof(n.Node_type));
        oss.write((char *)&n.Node_index, sizeof(n.Node_index));

        // inputs
        size_t num_inputs = n.inputs.size();
        oss.write((char *)&num_inputs, sizeof(num_inputs));
        for (auto &t : n.inputs)
            write_tensor(t);

        // outputs
        size_t num_outputs = n.outputs.size();
        oss.write((char *)&num_outputs, sizeof(num_outputs));
        for (auto &t : n.outputs)
            write_tensor(t);

        // params
        size_t num_params = n.params.size();
        oss.write((char *)&num_params, sizeof(num_params));
        for (auto &kv : n.params)
        {
            write_string(kv.first);
            oss.write((char *)&kv.second, sizeof(kv.second));
        }

        // fparams (float)
        {
            size_t num_fparams = n.fparams.size();
            oss.write((char *)&num_fparams, sizeof(num_fparams));
            for (auto &kv : n.fparams)
            {
                write_string(kv.first);
                oss.write((char *)&kv.second, sizeof(kv.second));
            }
        }
    };

    // graph_names
    write_string(model.target);

    size_t num_graphs = model.graph_names.size();
    oss.write((char *)&num_graphs, sizeof(num_graphs));
    for (auto &g : model.graph_names)
        write_string(g);

    // inputs
    size_t num_inputs = model.inputs.size();
    oss.write((char *)&num_inputs, sizeof(num_inputs));
    for (auto &t : model.inputs)
        write_tensor(t);

    // outputs
    size_t num_outputs = model.outputs.size();
    oss.write((char *)&num_outputs, sizeof(num_outputs));
    for (auto &t : model.outputs)
        write_tensor(t);

    // op_codes
    size_t num_ops = model.op_codes.size();
    oss.write((char *)&num_ops, sizeof(num_ops));
    for (auto &op : model.op_codes)
        write_string(op);

    // Node_List
    size_t num_nodes = model.Node_List.size();
    oss.write((char *)&num_nodes, sizeof(num_nodes));
    for (auto &n : model.Node_List)
        write_node(n);

    return oss.str();
}

// ================== 反序列化 ==================
AML_Model DeserializeModel(const std::string &bin)
{
    AML_Model model;
    std::istringstream iss(bin, std::ios::binary);

    auto read_string = [&](std::string &s)
    {
        size_t len;
        iss.read((char *)&len, sizeof(len));
        s.resize(len);
        iss.read(&s[0], len);
    };

    auto read_tensor = [&](AML_Tensor &t)
    {
        read_string(t.name);

        size_t dims_size;
        iss.read((char *)&dims_size, sizeof(dims_size));
        t.dims.resize(dims_size);
        iss.read((char *)t.dims.data(), dims_size * sizeof(int));

        iss.read((char *)&t.dtype, sizeof(t.dtype));
        iss.read((char *)&t.is_const, sizeof(t.is_const));
        iss.read((char *)&t.tensor_index, sizeof(t.tensor_index));

        // === 读取 zp ===
        size_t zp_size;
        iss.read((char *)&zp_size, sizeof(zp_size));
        t.zp.resize(zp_size);
        if (zp_size > 0)
            iss.read((char *)t.zp.data(), zp_size * sizeof(int));

        // === 读取 scale ===
        size_t scale_size;
        iss.read((char *)&scale_size, sizeof(scale_size));
        t.scale.resize(scale_size);
        if (scale_size > 0)
            iss.read((char *)t.scale.data(), scale_size * sizeof(float));

        size_t data_size;
        iss.read((char *)&data_size, sizeof(data_size));
        t.data.resize(data_size);
        if (data_size > 0)
            iss.read((char *)t.data.data(), data_size);

        iss.read((char *)&t.size, sizeof(t.size));
    };

    auto read_node = [&](AML_Node &n)
    {
        iss.read((char *)&n.Node_type, sizeof(n.Node_type));
        iss.read((char *)&n.Node_index, sizeof(n.Node_index));

        // inputs
        size_t num_inputs;
        iss.read((char *)&num_inputs, sizeof(num_inputs));
        n.inputs.resize(num_inputs);
        for (auto &t : n.inputs)
            read_tensor(t);

        // outputs
        size_t num_outputs;
        iss.read((char *)&num_outputs, sizeof(num_outputs));
        n.outputs.resize(num_outputs);
        for (auto &t : n.outputs)
            read_tensor(t);

        // params
        size_t num_params;
        iss.read((char *)&num_params, sizeof(num_params));
        for (size_t i = 0; i < num_params; ++i)
        {
            std::string key;
            int value;
            read_string(key);
            iss.read((char *)&value, sizeof(value));
            n.params[key] = value;
        }

        // fparams (float)
        size_t num_fparams;
        iss.read((char *)&num_fparams, sizeof(num_fparams));
        for (size_t i = 0; i < num_fparams; ++i)
        {
            std::string key;
            float value;
            read_string(key);
            iss.read((char *)&value, sizeof(value));
            n.fparams[key] = value;
        }
    };

    // target
    read_string(model.target);

    // graph_names
    size_t num_graphs;
    iss.read((char *)&num_graphs, sizeof(num_graphs));
    model.graph_names.resize(num_graphs);
    for (auto &g : model.graph_names)
        read_string(g);

    // inputs
    size_t num_inputs;
    iss.read((char *)&num_inputs, sizeof(num_inputs));
    model.inputs.resize(num_inputs);
    for (auto &t : model.inputs)
        read_tensor(t);

    // outputs
    size_t num_outputs;
    iss.read((char *)&num_outputs, sizeof(num_outputs));
    model.outputs.resize(num_outputs);
    for (auto &t : model.outputs)
        read_tensor(t);

    // op_codes
    size_t num_ops;
    iss.read((char *)&num_ops, sizeof(num_ops));
    model.op_codes.resize(num_ops);
    for (auto &op : model.op_codes)
        read_string(op);

    // Node_List
    size_t num_nodes;
    iss.read((char *)&num_nodes, sizeof(num_nodes));
    model.Node_List.resize(num_nodes);
    for (auto &n : model.Node_List)
        read_node(n);

    return model;
}

namespace {

std::string FormatIntVec(const std::vector<int>& values) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

std::string FormatFloatVec(const std::vector<float>& values) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

std::string FormatStringVec(const std::vector<std::string>& values) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

}  // namespace

void PrintAML_Tensor(const AML_Tensor& tensor, const char* indent) {
  const char* prefix = (indent != nullptr) ? indent : "";
  const size_t payload_bytes =
      tensor.size > 0 ? tensor.size : tensor.data.size();
  ADLA_LOGI("%stensor name=\"%s\" index=%d dtype=%d is_const=%d size=%zu",
            prefix, tensor.name.c_str(), tensor.tensor_index, tensor.dtype,
            tensor.is_const ? 1 : 0, payload_bytes);
  ADLA_LOGI("%s  dims=%s zp=%s scale=%s", prefix,
            FormatIntVec(tensor.dims).c_str(),
            FormatIntVec(tensor.zp).c_str(),
            FormatFloatVec(tensor.scale).c_str());
}

void PrintAML_Node(const AML_Node& node, const char* indent) {
  const char* prefix = (indent != nullptr) ? indent : "";
  ADLA_LOGI("%snode index=%d op_type=%d", prefix, node.Node_index,
            node.Node_type);
  if (!node.params.empty()) {
    for (const auto& kv : node.params) {
      ADLA_LOGI("%s  param[%s]=%d", prefix, kv.first.c_str(), kv.second);
    }
  }
  if (!node.fparams.empty()) {
    for (const auto& kv : node.fparams) {
      ADLA_LOGI("%s  fparam[%s]=%g", prefix, kv.first.c_str(), kv.second);
    }
  }
  ADLA_LOGI("%s  inputs (%zu):", prefix, node.inputs.size());
  for (size_t i = 0; i < node.inputs.size(); ++i) {
    std::ostringstream line;
    line << prefix << "    in[" << i << "] ";
    PrintAML_Tensor(node.inputs[i], line.str().c_str());
  }
  ADLA_LOGI("%s  outputs (%zu):", prefix, node.outputs.size());
  for (size_t i = 0; i < node.outputs.size(); ++i) {
    std::ostringstream line;
    line << prefix << "    out[" << i << "] ";
    PrintAML_Tensor(node.outputs[i], line.str().c_str());
  }
}

void PrintAML_Model(const AML_Model& model, int graph_index) {
  ADLA_LOGI("========== AML Graph #%d ==========", graph_index);
  ADLA_LOGI("target=\"%s\"", model.target.c_str());
  if (!model.graph_names.empty()) {
    ADLA_LOGI("graph_names=%s", FormatStringVec(model.graph_names).c_str());
  }
  if (!model.op_codes.empty()) {
    ADLA_LOGI("op_codes=%s", FormatStringVec(model.op_codes).c_str());
  }

  ADLA_LOGI("--- graph inputs (%zu) ---", model.inputs.size());
  for (size_t i = 0; i < model.inputs.size(); ++i) {
    std::ostringstream line;
    line << "  in[" << i << "] ";
    PrintAML_Tensor(model.inputs[i], line.str().c_str());
  }

  ADLA_LOGI("--- graph outputs (%zu) ---", model.outputs.size());
  for (size_t i = 0; i < model.outputs.size(); ++i) {
    std::ostringstream line;
    line << "  out[" << i << "] ";
    PrintAML_Tensor(model.outputs[i], line.str().c_str());
  }

  ADLA_LOGI("--- nodes (%zu) ---", model.Node_List.size());
  for (size_t i = 0; i < model.Node_List.size(); ++i) {
    std::ostringstream line;
    line << "  [" << i << "] ";
    PrintAML_Node(model.Node_List[i], line.str().c_str());
  }
  ADLA_LOGI("========== end AML Graph #%d ==========", graph_index);
}
