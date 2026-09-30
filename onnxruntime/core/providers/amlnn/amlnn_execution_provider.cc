#include "amlnn_execution_provider.h"

// Amlogic NPU execution provider implementation.
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cerrno>
#include <sys/stat.h>

#include "amlnn_nnsdk2.h"
#include "amlnn_logging.h"
#include "core/providers/amlogic/amlogic_model_info.h"
#include "core/framework/compute_capability.h"
#include "core/session/onnxruntime_cxx_api.h"
#if defined(ORT_AMLNN_USE_AMLOGIC_CONVERTER)
#include "aml_adla_bridge_c.h"
#endif

namespace onnxruntime {
namespace {

std::string ShapeString(const std::vector<int64_t>& shape) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) os << ",";
    os << shape[i];
  }
  os << "]";
  return os.str();
}

std::string AttrString(const amlnn_tensor_attr& attr) {
  std::ostringstream os;
  os << "name=" << attr.name << " index=" << attr.index
     << " format=" << get_format_string(attr.fmt)
     << " type=" << get_type_string(attr.type) << " dims=[";
  for (uint32_t i = 0; i < attr.n_dims; ++i) {
    if (i != 0) os << ",";
    os << attr.dims[i];
  }
  os << "] elements=" << attr.n_elems << " bytes=" << attr.size;
  return os.str();
}

#if defined(ORT_AMLNN_USE_AMLOGIC_CONVERTER)
bool NonemptyFile(const std::string& path) {
  struct stat st{};
  return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0;
}

bool EnsureDirectory(const std::string& path) {
  if (path.empty()) return false;
  if (mkdir(path.c_str(), 0777) == 0 || errno == EEXIST) return true;
  return false;
}

std::string PathDirname(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? std::string() : path.substr(0, slash);
}

std::string PathStem(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  const size_t begin = slash == std::string::npos ? 0 : slash + 1;
  const size_t dot = path.find_last_of('.');
  const size_t end = dot == std::string::npos || dot < begin ? path.size() : dot;
  return path.substr(begin, end - begin);
}

common::Status PrepareAmlnnModel(const GraphViewer& graph_viewer, const logging::Logger& logger,
                                 std::string& model_path) {
  auto ExceptionStatus = [](const char* phase, const std::exception& exception) {
    const std::string message = std::string("Amlogic conversion failed during ") + phase + ": " + exception.what();
    LOGE << message << "\n";
    return common::Status(common::ONNXRUNTIME, common::FAIL, message);
  };
  try {
  const char* configured_path = std::getenv("AMLNN_MODEL_PATH");
  if (configured_path != nullptr && configured_path[0] != '\0') {
    model_path = configured_path;
    return common::Status::OK();
  }

  const auto& onnx_fs_path = graph_viewer.ModelPath();
  const std::string onnx_path(onnx_fs_path.c_str());
  if (onnx_path.empty()) {
    return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                          "AMLNN_MODEL_PATH is required when the model path is unavailable");
  }

  const char* output_env = std::getenv("ORT_AMLOGIC_CONVERT_OUTPUT_DIR");
  const std::string output_dir = output_env != nullptr && output_env[0] != '\0'
                                     ? output_env
                                     : PathDirname(onnx_path) + "/amlogic_runtime_artifacts";
  if (!EnsureDirectory(output_dir)) {
    return common::Status(common::ONNXRUNTIME, common::FAIL,
                          "cannot create Amlogic conversion directory: " + output_dir);
  }

  const std::string model_name = PathStem(onnx_path);
  const std::string adla_path = output_dir + "/" + model_name + ".adla";
  if (NonemptyFile(adla_path)) {
    model_path = adla_path;
    LOGI << "Using existing ADLA: " << model_path << "\n";
    return common::Status::OK();
  }

  const std::string json_path = output_dir + "/amlogic_onnx_nhwc_ops.json";
  setenv("ORT_AMLOGIC_EXPORT_MODE", "reference_style", 1);
  LOGI << "Amlogic conversion: exporting reference-style JSON to " << json_path << "\n";
  auto dump_status = amlogic::DumpGraphInfoAsJson(graph_viewer, json_path, logger);
  if (!dump_status.IsOK()) {
    return dump_status;
  }
  std::ifstream json_file(json_path, std::ios::binary);
  if (!json_file) {
    return common::Status(common::ONNXRUNTIME, common::FAIL,
                          "Amlogic exporter did not create: " + json_path);
  }
  const std::string json((std::istreambuf_iterator<char>(json_file)), {});
  if (json.empty()) {
    return common::Status(common::ONNXRUNTIME, common::FAIL,
                          "Amlogic exporter produced an empty JSON file: " + json_path);
  }

  const char* target_env = std::getenv("AMLOGIC_ADLA_TARGET");
  const std::string target = target_env != nullptr && target_env[0] != '\0' ? target_env : "a9";
  setenv("AML_ADLA_BRIDGE_DISABLE_LAYOUT_FOLDS", "1", 0);
  std::vector<char> result_path(4096);
  std::vector<char> error_message(4096);
  const int result = aml_compile_ops_json_to_adla(
      json.c_str(), output_dir.c_str(), model_name.c_str(), target.c_str(),
      result_path.data(), result_path.size(), error_message.data(), error_message.size());
  if (result != 0) {
    return common::Status(common::ONNXRUNTIME, common::FAIL,
                          std::string("Amlogic ADLA conversion failed: ") + error_message.data());
  }
  model_path = result_path.data();
  if (model_path.empty()) {
    return common::Status(common::ONNXRUNTIME, common::FAIL,
                          "Amlogic ADLA conversion returned an empty path");
  }
  LOGI << "Converted ONNX to ADLA during Session initialization: " << model_path << "\n";
  return common::Status::OK();
  } catch (const std::exception& exception) {
    return ExceptionStatus("JSON export or ADLA compilation", exception);
  } catch (...) {
    const std::string message = "Amlogic conversion failed during JSON export or ADLA compilation: unknown exception";
    LOGE << message << "\n";
    return common::Status(common::ONNXRUNTIME, common::FAIL, message);
  }
}
#endif

size_t TypeBytes(amlnn_tensor_type type) {
  switch (type) {
    case AMLNN_TENSOR_FLOAT32:
    case AMLNN_TENSOR_INT32:
    case AMLNN_TENSOR_UINT32:
      return 4;
    case AMLNN_TENSOR_FLOAT16:
    case AMLNN_TENSOR_BFLOAT16:
    case AMLNN_TENSOR_INT16:
    case AMLNN_TENSOR_UINT16:
      return 2;
    case AMLNN_TENSOR_INT8:
    case AMLNN_TENSOR_UINT8:
    case AMLNN_TENSOR_BOOL:
      return 1;
    case AMLNN_TENSOR_INT64:
    case AMLNN_TENSOR_UINT64:
      return 8;
    default:
      return 0;
  }
}

bool IsNchwNhwcPair(const std::vector<int64_t>& ort_shape, const amlnn_tensor_attr& attr) {
  return ort_shape.size() == 4 && attr.n_dims == 4 &&
         ort_shape[0] == static_cast<int64_t>(attr.dims[0]) &&
         ort_shape[1] == static_cast<int64_t>(attr.dims[3]) &&
         ort_shape[2] == static_cast<int64_t>(attr.dims[1]) &&
         ort_shape[3] == static_cast<int64_t>(attr.dims[2]);
}

void NchwToNhwc(const uint8_t* src, uint8_t* dst, const std::vector<int64_t>& shape, size_t element_size) {
  const size_t n_size = static_cast<size_t>(shape[0]);
  const size_t c_size = static_cast<size_t>(shape[1]);
  const size_t h_size = static_cast<size_t>(shape[2]);
  const size_t w_size = static_cast<size_t>(shape[3]);
  for (size_t n = 0; n < n_size; ++n)
    for (size_t h = 0; h < h_size; ++h)
      for (size_t w = 0; w < w_size; ++w)
        for (size_t c = 0; c < c_size; ++c) {
          const size_t src_index = ((n * c_size + c) * h_size + h) * w_size + w;
          const size_t dst_index = ((n * h_size + h) * w_size + w) * c_size + c;
          std::memcpy(dst + dst_index * element_size, src + src_index * element_size, element_size);
        }
}

void NhwcToNchw(const uint8_t* src, uint8_t* dst, const std::vector<int64_t>& shape, size_t element_size) {
  const size_t n_size = static_cast<size_t>(shape[0]);
  const size_t h_size = static_cast<size_t>(shape[1]);
  const size_t w_size = static_cast<size_t>(shape[2]);
  const size_t c_size = static_cast<size_t>(shape[3]);
  for (size_t n = 0; n < n_size; ++n)
    for (size_t h = 0; h < h_size; ++h)
      for (size_t w = 0; w < w_size; ++w)
        for (size_t c = 0; c < c_size; ++c) {
          const size_t src_index = ((n * h_size + h) * w_size + w) * c_size + c;
          const size_t dst_index = ((n * c_size + c) * h_size + h) * w_size + w;
          std::memcpy(dst + dst_index * element_size, src + src_index * element_size, element_size);
        }
}

bool IsTypeCompatible(ONNXTensorElementDataType ort_type, amlnn_tensor_type aml_type) {
  switch (ort_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return aml_type == AMLNN_TENSOR_FLOAT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return aml_type == AMLNN_TENSOR_FLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return aml_type == AMLNN_TENSOR_BFLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return aml_type == AMLNN_TENSOR_INT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return aml_type == AMLNN_TENSOR_UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return aml_type == AMLNN_TENSOR_INT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return aml_type == AMLNN_TENSOR_UINT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return aml_type == AMLNN_TENSOR_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return aml_type == AMLNN_TENSOR_UINT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return aml_type == AMLNN_TENSOR_INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return aml_type == AMLNN_TENSOR_UINT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return aml_type == AMLNN_TENSOR_BOOL;
    default:
      return false;
  }
}

}  // namespace

AmlnnExecutionProvider::AmlnnExecutionProvider() : IExecutionProvider{"Amlnn"} {
}

std::vector<std::unique_ptr<ComputeCapability>> AmlnnExecutionProvider::GetCapability(
    const GraphViewer& g, const IKernelLookup&, const GraphOptimizerRegistry&,
    IResourceAccountant*) const {
  std::vector<std::unique_ptr<ComputeCapability>> r;
  if (g.IsSubgraph() || g.NumberOfNodes() == 0) return r;
  auto sg = std::make_unique<IndexedSubGraph>();
  for (auto i : g.GetNodesInTopologicalOrder()) sg->nodes.push_back(i);
  auto m = std::make_unique<IndexedSubGraph::MetaDef>();
  m->name = "AMLNN";
  m->domain = kMSDomain;
  m->since_version = 1;
  for (auto* x : g.GetInputs()) m->inputs.push_back(x->Name());
  for (auto* x : g.GetOutputs()) m->outputs.push_back(x->Name());
  sg->SetMetaDef(std::move(m));
  r.push_back(std::make_unique<ComputeCapability>(std::move(sg)));
  return r;
}

common::Status AmlnnExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& groups,
                                               std::vector<NodeComputeInfo>& funcs) {
  if (groups.empty()) {
    const std::string error = "AMLNN Compile received no fused graph";
    LOGE << error << "\n";
    return common::Status(common::ONNXRUNTIME, common::FAIL, error);
  }

  std::string model_path;
#if defined(ORT_AMLNN_USE_AMLOGIC_CONVERTER)
  auto prepare_status = PrepareAmlnnModel(groups.front().filtered_graph, *GetLogger(), model_path);
  if (!prepare_status.IsOK()) return prepare_status;
#else
  const char* configured_model_path = std::getenv("AMLNN_MODEL_PATH");
  if (configured_model_path == nullptr || configured_model_path[0] == '\0') {
    return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                          "AMLNN_MODEL_PATH is required");
  }
  model_path = configured_model_path;
#endif
  std::ifstream f(model_path, std::ios::binary);
  if (!f) {
    const std::string error = "cannot open AMLNN model: " + model_path;
    LOGE << error << "\n";
    return common::Status(common::ONNXRUNTIME, common::FAIL, error);
  }
  std::vector<uint8_t> model((std::istreambuf_iterator<char>(f)), {});
  if (model.empty()) {
    const std::string error = "AMLNN model is empty: " + std::string(model_path);
    LOGE << error << "\n";
    return common::Status(common::ONNXRUNTIME, common::FAIL, error);
  }
  LOGI << "Compile: groups=" << groups.size()
       << " model=" << model_path << " model_bytes=" << model.size()
       << "\n";
  for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
    auto runtime = std::make_shared<AmlnnNnsdk2>();
    auto st = runtime->Load();
    if (!st.IsOK()) return st;
    st = runtime->InitModel(model);
    if (!st.IsOK()) return st;
    LOGI << "Group " << group_index << ": inputs=" << runtime->Inputs().size()
         << " outputs=" << runtime->Outputs().size() << "\n";
    for (const auto& attr : runtime->Inputs()) LOGI << "ADLA input: " << AttrString(attr) << "\n";
    for (const auto& attr : runtime->Outputs()) LOGI << "ADLA output: " << AttrString(attr) << "\n";
    NodeComputeInfo ci;
    ci.create_state_func = [runtime](ComputeContext*, FunctionState* s) {
      *s = runtime.get();
      return 0;
    };
    ci.release_state_func = [](FunctionState) {};
    ci.compute_func = [runtime](FunctionState, const OrtApi*, OrtKernelContext* c) {
      Ort::KernelContext ctx(c);
      if (ctx.GetInputCount() != runtime->Inputs().size()) {
        const std::string error = "AMLNN input count mismatch: ORT=" + std::to_string(ctx.GetInputCount()) +
                                  " ADLA=" + std::to_string(runtime->Inputs().size());
        LOGE << error << "\n";
        return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, error);
      }

      std::vector<void*> inputs;
      std::vector<uint32_t> input_sizes;
      std::vector<std::vector<uint8_t>> converted_inputs;
      converted_inputs.reserve(ctx.GetInputCount());
      for (size_t i = 0; i < ctx.GetInputCount(); ++i) {
        auto value = ctx.GetInput(i);
        auto info = value.GetTensorTypeAndShapeInfo();
        const auto ort_shape = info.GetShape();
        const auto& attr = runtime->Inputs()[i];
        LOGD << "ORT input " << i << ": shape=" << ShapeString(ort_shape)
             << " type=" << info.GetElementType() << " bytes=" << value.GetTensorSizeInBytes() << "\n";

        if (!IsTypeCompatible(info.GetElementType(), attr.type)) {
          const std::string error = "AMLNN input type mismatch at index " + std::to_string(i) +
                                    ": ORT type=" + std::to_string(info.GetElementType()) +
                                    " ADLA=" + AttrString(attr);
          LOGE << error << "\n";
          return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, error);
        }
        const bool transpose = attr.fmt == AMLNN_TENSOR_NHWC && IsNchwNhwcPair(ort_shape, attr);
        bool direct_shape = attr.n_dims != 4 && ort_shape.size() == attr.n_dims;
        for (size_t d = 0; direct_shape && d < ort_shape.size(); ++d) {
          direct_shape = ort_shape[d] < 0 || static_cast<uint32_t>(ort_shape[d]) == attr.dims[d];
        }
        if (!direct_shape && !transpose) {
          const std::string error = "AMLNN input shape mismatch at index " + std::to_string(i) +
                                    ": ORT=" + ShapeString(ort_shape) + " ADLA=" + AttrString(attr);
          LOGE << error << "\n";
          return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, error);
        }
        const size_t element_size = TypeBytes(attr.type);
        if (element_size == 0 || value.GetTensorSizeInBytes() != attr.size) {
          const std::string error = "AMLNN input byte size mismatch at index " + std::to_string(i) +
                                    ": ORT=" + std::to_string(value.GetTensorSizeInBytes()) +
                                    " ADLA=" + AttrString(attr);
          LOGE << error << "\n";
          return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, error);
        }
        if (transpose) {
          converted_inputs.emplace_back(attr.size);
          NchwToNhwc(static_cast<const uint8_t*>(value.GetTensorRawData()), converted_inputs.back().data(),
                     ort_shape, element_size);
          inputs.push_back(converted_inputs.back().data());
          LOGD << "Input " << i << " converted NCHW -> NHWC\n";
        } else {
          inputs.push_back(const_cast<void*>(value.GetTensorRawData()));
        }
        input_sizes.push_back(static_cast<uint32_t>(value.GetTensorSizeInBytes()));
      }

      std::vector<std::vector<uint8_t>> outputs;
      auto status = runtime->Run(inputs, input_sizes, outputs);
      if (!status.IsOK()) return status;
      if (outputs.size() != ctx.GetOutputCount()) {
        const std::string error = "AMLNN output count mismatch: ORT=" + std::to_string(ctx.GetOutputCount()) +
                                  " ADLA=" + std::to_string(outputs.size());
        LOGE << error << "\n";
        return common::Status(common::ONNXRUNTIME, common::FAIL, error);
      }
      for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& attr = runtime->Outputs()[i];
        std::vector<int64_t> adla_shape(attr.dims, attr.dims + attr.n_dims);
        const bool transpose = attr.fmt == AMLNN_TENSOR_NHWC && attr.n_dims == 4;
        std::vector<int64_t> ort_shape = adla_shape;
        if (transpose) ort_shape = {adla_shape[0], adla_shape[3], adla_shape[1], adla_shape[2]};
        auto value = ctx.GetOutput(i, ort_shape.data(), ort_shape.size());
        auto info = value.GetTensorTypeAndShapeInfo();
        LOGD << "ORT output " << i << ": shape=" << ShapeString(ort_shape)
             << " type=" << info.GetElementType() << " bytes=" << value.GetTensorSizeInBytes() << "\n";
        if (!IsTypeCompatible(info.GetElementType(), attr.type)) {
          const std::string error = "AMLNN output type mismatch at index " + std::to_string(i) +
                                    ": ORT type=" + std::to_string(info.GetElementType()) +
                                    " ADLA=" + AttrString(attr);
          LOGE << error << "\n";
          return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, error);
        }
        const size_t element_size = TypeBytes(attr.type);
        if (element_size == 0 || value.GetTensorSizeInBytes() != attr.size || outputs[i].size() != attr.size) {
          const std::string error = "AMLNN output byte size mismatch at index " + std::to_string(i) +
                                    ": ORT=" + std::to_string(value.GetTensorSizeInBytes()) +
                                    " runtime=" + std::to_string(outputs[i].size()) + " ADLA=" + AttrString(attr);
          LOGE << error << "\n";
          return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, error);
        }
        auto* destination = static_cast<uint8_t*>(const_cast<void*>(value.GetTensorRawData()));
        if (transpose) {
          NhwcToNchw(outputs[i].data(), destination, adla_shape, element_size);
          LOGD << "Output " << i << " converted NHWC -> NCHW\n";
        } else {
          std::memcpy(destination, outputs[i].data(), outputs[i].size());
        }
      }
      LOGD << "Compute completed successfully\n";
      return common::Status::OK();
    };
    funcs.push_back(std::move(ci));
  }
  return common::Status::OK();
}
}  // namespace onnxruntime
