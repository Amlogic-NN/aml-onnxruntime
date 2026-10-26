#include "amlnn_nnsdk2.h"
#include "amlnn_logging.h"

#include <dlfcn.h>

#include <fstream>
#include <limits>
#include <string>
#include <iostream>
#include <sstream>

namespace onnxruntime {
AmlnnNnsdk2::~AmlnnNnsdk2() {
  Reset();
}

void AmlnnNnsdk2::Reset() {
  if (ctx_ != nullptr && destroy_ != nullptr) {
    LOGD << "[AMLNN NNSDK2] amlnn_destroy ctx=" << ctx_ << "\n";
    const int rc = destroy_(ctx_);
    LOGD << "[AMLNN NNSDK2] amlnn_destroy rc=" << rc << "\n";
  }
  ctx_ = nullptr;
  if (so_ != nullptr) dlclose(so_);
  so_ = nullptr;
  init_ = nullptr;
  query_ = nullptr;
  inputs_set_ = nullptr;
  run_ = nullptr;
  outputs_get_ = nullptr;
  destroy_ = nullptr;
  in_attrs_.clear();
  out_attrs_.clear();
}

common::Status AmlnnNnsdk2::Load() {
  if (so_) {
    LOGE << "NNSDK2 library is already loaded\n";
    return common::Status(common::ONNXRUNTIME, common::FAIL, "NNSDK2 library is already loaded");
  }
  LOGD << "[AMLNN NNSDK2] Loading library: libnnsdk.so\n";
  so_ = dlopen("libnnsdk.so", RTLD_NOW | RTLD_LOCAL);
  if (!so_) {
    const char* e = dlerror();
    const std::string error = e ? e : "unable to load NNSDK2 library";
    LOGE << error << "\n";
    return common::Status(common::ONNXRUNTIME, common::FAIL, error);
  }
#define AMLNN_LOAD_SYMBOL(member, symbol)                                   \
  do {                                                                      \
    member = reinterpret_cast<decltype(member)>(dlsym(so_, #symbol));       \
    if (member == nullptr) {                                                \
      const std::string message = std::string("missing symbol ") + #symbol; \
      Reset();                                                              \
      LOGE << message << "\n";                                              \
      return common::Status(common::ONNXRUNTIME, common::FAIL, message);    \
    }                                                                       \
  } while (0)

  AMLNN_LOAD_SYMBOL(init_, amlnn_init);
  AMLNN_LOAD_SYMBOL(query_, amlnn_query);
  AMLNN_LOAD_SYMBOL(inputs_set_, amlnn_inputs_set);
  AMLNN_LOAD_SYMBOL(run_, amlnn_run);
  AMLNN_LOAD_SYMBOL(outputs_get_, amlnn_outputs_get);
  AMLNN_LOAD_SYMBOL(destroy_, amlnn_destroy);

#undef AMLNN_LOAD_SYMBOL
  LOGD << "[AMLNN NNSDK2] All required symbols loaded\n";
  return common::Status::OK();
}

common::Status AmlnnNnsdk2::InitModel(const std::vector<uint8_t>& model) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!init_) {
    LOGE << "libnnsdk.so is not loaded\n";
    return common::Status(common::ONNXRUNTIME, common::FAIL, "libnnsdk.so is not loaded");
  }
  if (ctx_) {
    LOGE << "NNSDK2 model is already initialized\n";
    return common::Status(common::ONNXRUNTIME, common::FAIL, "NNSDK2 model is already initialized");
  }
  if (model.empty()) {
    LOGE << "AMLNN model is empty\n";
    return common::Status(common::ONNXRUNTIME, common::FAIL, "AMLNN model is empty");
  }
  if (model.size() > std::numeric_limits<uint32_t>::max()) {
    LOGE << "AMLNN model is too large\n";
    return common::Status(common::ONNXRUNTIME, common::FAIL, "AMLNN model is too large");
  }
  amlnn_init_config c{};
  c.backend_type = AMLNN_BACKEND_ADLA_NPU;
  LOGD << "[AMLNN NNSDK2] amlnn_init model_bytes=" << model.size() << " backend=ADLA_NPU\n";
  int r = init_(&ctx_, const_cast<uint8_t*>(model.data()), static_cast<uint32_t>(model.size()), &c);
  LOGD << "[AMLNN NNSDK2] amlnn_init rc=" << r << " ctx=" << ctx_ << "\n";
  if (r < 0) {
    const std::string error = "amlnn_init failed: " + std::to_string(r);
    LOGE << error << "\n";
    return common::Status(common::ONNXRUNTIME, common::FAIL, error);
  }
  amlnn_input_output_num n{};
  r = query_(ctx_, AMLNN_QUERY_IN_OUT_NUM, &n, sizeof(n));
  LOGD << "[AMLNN NNSDK2] query in/out rc=" << r << " inputs=" << n.n_input << " outputs=" << n.n_output << "\n";
  if (r < 0) {
    const std::string error = "amlnn_query failed: " + std::to_string(r);
    LOGE << error << "\n";
    return common::Status(common::ONNXRUNTIME, common::FAIL, error);
  }
  in_attrs_.resize(n.n_input);
  out_attrs_.resize(n.n_output);
  for (uint32_t i = 0; i < n.n_input; i++) {
    in_attrs_[i].index = i;
    const int qr = query_(ctx_, AMLNN_QUERY_INPUT_ATTR, &in_attrs_[i], sizeof(in_attrs_[i]));
    LOGD << "[AMLNN NNSDK2] query input attr index=" << i << " rc=" << qr;
    if (qr < 0) {
      LOGD << "\n";
      LOGE << "query input attr failed\n";
      return common::Status(common::ONNXRUNTIME, common::FAIL, "query input attr failed");
    }
    LOGD << " name=" << in_attrs_[i].name << " format=" << get_format_string(in_attrs_[i].fmt)
         << " type=" << get_type_string(in_attrs_[i].type) << " dims=" << in_attrs_[i].n_dims << " [";
    for (uint32_t d = 0; d < in_attrs_[i].n_dims; ++d) {
      if (d) {
        LOGD << ",";
      }
      LOGD << in_attrs_[i].dims[d];
    }
    LOGD << "] bytes=" << in_attrs_[i].size << "\n";
  }
  for (uint32_t i = 0; i < n.n_output; i++) {
    out_attrs_[i].index = i;
    const int qr = query_(ctx_, AMLNN_QUERY_OUTPUT_ATTR, &out_attrs_[i], sizeof(out_attrs_[i]));
    LOGD << "[AMLNN NNSDK2] query output attr index=" << i << " rc=" << qr;
    if (qr < 0) {
      LOGD << "\n";
      LOGE << "query output attr failed\n";
      return common::Status(common::ONNXRUNTIME, common::FAIL, "query output attr failed");
    }
    LOGD << " name=" << out_attrs_[i].name << " format=" << get_format_string(out_attrs_[i].fmt)
         << " type=" << get_type_string(out_attrs_[i].type) << " dims=" << out_attrs_[i].n_dims << " [";
    for (uint32_t d = 0; d < out_attrs_[i].n_dims; ++d) {
      if (d) {
        LOGD << ",";
      }
      LOGD << out_attrs_[i].dims[d];
    }
    LOGD << "] bytes=" << out_attrs_[i].size << "\n";
  }
  LOGD << "[AMLNN NNSDK2] Model attributes queried successfully\n";
  return common::Status::OK();
}

common::Status AmlnnNnsdk2::Run(const std::vector<void*>& bufs, const std::vector<uint32_t>& sizes,
                                std::vector<std::vector<uint8_t>>& outs) {
  std::lock_guard<std::mutex> l(mutex_);
  if (ctx_ == nullptr) {
    LOGE << "NNSDK2 model is not initialized\n";
    return common::Status(common::ONNXRUNTIME, common::FAIL, "NNSDK2 model is not initialized");
  }
  if (bufs.size() != in_attrs_.size() || sizes.size() != bufs.size()) {
    LOGE << "input count mismatch\n";
    return common::Status(common::ONNXRUNTIME, common::FAIL, "input count mismatch");
  }
  for (size_t i = 0; i < bufs.size(); ++i) {
    if (bufs[i] == nullptr) {
      LOGE << "input buffer is null\n";
      return common::Status(common::ONNXRUNTIME, common::FAIL, "input buffer is null");
    }
    if (sizes[i] < in_attrs_[i].size) {
      {
        const std::string error = "input buffer is smaller than AMLNN tensor size at index " + std::to_string(i) + ": provided=" + std::to_string(sizes[i]) + " required=" + std::to_string(in_attrs_[i].size);
        LOGE << error << "\n";
        return common::Status(common::ONNXRUNTIME, common::FAIL, error);
      }
    }
    LOGD << "[AMLNN NNSDK2] input " << i << " buffer=" << bufs[i]
         << " bytes=" << sizes[i] << " required=" << in_attrs_[i].size << "\n";
  }
  if (bufs.size() > std::numeric_limits<uint32_t>::max()) {
    LOGE << "too many AMLNN inputs\n";
    return common::Status(common::ONNXRUNTIME, common::FAIL, "too many AMLNN inputs");
  }
  std::vector<amlnn_input> in(bufs.size());
  for (size_t i = 0; i < in.size(); i++) {
    in[i].index = static_cast<uint32_t>(i);
    in[i].buf = bufs[i];
    in[i].size = sizes[i];
  }
  LOGD << "[AMLNN NNSDK2] amlnn_inputs_set count=" << in.size() << "\n";
  int r = inputs_set_(ctx_, static_cast<uint32_t>(in.size()), in.data());
  LOGD << "[AMLNN NNSDK2] amlnn_inputs_set rc=" << r << "\n";
  if (r < 0) {
    const std::string error = "amlnn_inputs_set failed: " + std::to_string(r);
    LOGE << error << "\n";
    return common::Status(common::ONNXRUNTIME, common::FAIL, error);
  }
  LOGD << "[AMLNN NNSDK2] amlnn_run\n";
  r = run_(ctx_, nullptr);
  LOGD << "[AMLNN NNSDK2] amlnn_run rc=" << r << "\n";
  if (r < 0) {
    const std::string error = "amlnn_run failed: " + std::to_string(r);
    LOGE << error << "\n";
    return common::Status(common::ONNXRUNTIME, common::FAIL, error);
  }
  outs.resize(out_attrs_.size());
  if (outs.size() > std::numeric_limits<uint32_t>::max()) {
    LOGE << "too many AMLNN outputs\n";
    return common::Status(common::ONNXRUNTIME, common::FAIL, "too many AMLNN outputs");
  }
  std::vector<amlnn_output> o(outs.size());
  for (size_t i = 0; i < o.size(); i++) {
    o[i].index = static_cast<uint32_t>(i);
    o[i].is_float = 0;
    // NNSDK2 owns the output storage. amlnn_outputs_get fills buf and size.
    o[i].buf = nullptr;
    o[i].size = 0;
    LOGD << "[AMLNN NNSDK2] request output " << i
         << " is_float=" << o[i].is_float << "\n";
  }
  LOGD << "[AMLNN NNSDK2] amlnn_outputs_get count=" << o.size() << "\n";
  r = outputs_get_(ctx_, static_cast<uint32_t>(o.size()), o.data());
  LOGD << "[AMLNN NNSDK2] amlnn_outputs_get rc=" << r << "\n";
  if (r < 0) {
    const std::string error = "amlnn_outputs_get failed: " + std::to_string(r);
    LOGE << error << "\n";
    return common::Status(common::ONNXRUNTIME, common::FAIL, error);
  }
  for (size_t i = 0; i < o.size(); ++i) {
    LOGD << "[AMLNN NNSDK2] returned output " << i << " buffer=" << o[i].buf
         << " bytes=" << o[i].size << " expected=" << out_attrs_[i].size << "\n";
    if (o[i].buf == nullptr) {
      {
        const std::string error = "amlnn_outputs_get returned a null buffer at index " + std::to_string(i);
        LOGE << error << "\n";
        return common::Status(common::ONNXRUNTIME, common::FAIL, error);
      }
    }
    if (o[i].size != out_attrs_[i].size) {
      {
        const std::string error = "amlnn_outputs_get returned an unexpected size at index " + std::to_string(i) + ": returned=" + std::to_string(o[i].size) + " expected=" + std::to_string(out_attrs_[i].size);
        LOGE << error << "\n";
        return common::Status(common::ONNXRUNTIME, common::FAIL, error);
      }
    }
    const auto* begin = static_cast<const uint8_t*>(o[i].buf);
    size_t nonzero_bytes = 0;
    for (size_t j = 0; j < o[i].size; ++j) {
      if (begin[j] != 0) ++nonzero_bytes;
    }
    LOGD << "[AMLNN NNSDK2] returned output " << i
         << " nonzero_bytes=" << nonzero_bytes << "/" << o[i].size << "\n";
    outs[i].assign(begin, begin + o[i].size);
  }
  return common::Status::OK();
}
}  // namespace onnxruntime
