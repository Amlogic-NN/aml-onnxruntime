#pragma once
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "core/common/status.h"
#include "nnsdk2.h"

namespace onnxruntime {
class AmlnnNnsdk2 {
 public:
  ~AmlnnNnsdk2();
  common::Status Load();
  common::Status InitModel(const std::vector<uint8_t>& model);
  common::Status Run(const std::vector<void*>& inputs, const std::vector<uint32_t>& input_sizes,
                     std::vector<std::vector<uint8_t>>& outputs);

 private:
  void Reset();

  void* so_{nullptr};
  void* ctx_{nullptr};
  int (*init_)(void**, void*, uint32_t, amlnn_init_config*){nullptr};
  int (*query_)(void*, amlnn_query_cmd, void*, uint32_t){nullptr};
  int (*inputs_set_)(void*, uint32_t, amlnn_input[]){nullptr};
  int (*run_)(void*, amlnn_run_config*){nullptr};
  int (*outputs_get_)(void*, uint32_t, amlnn_output[]){nullptr};
  int (*destroy_)(void*){nullptr};
  std::vector<amlnn_tensor_attr> in_attrs_, out_attrs_;
  std::mutex mutex_;

 public:
  const std::vector<amlnn_tensor_attr>& Inputs() const {
    return in_attrs_;
  }

  const std::vector<amlnn_tensor_attr>& Outputs() const {
    return out_attrs_;
  }
};
}  // namespace onnxruntime
