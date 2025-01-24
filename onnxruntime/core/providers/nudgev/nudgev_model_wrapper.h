// nudgev_model_wrapper.h
#pragma once

#include <map>
#include <string>
#include <vector>
#include "core/framework/node_unit.h"
#include "core/graph/graph_viewer.h"
#include "core/common/logging/logging.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace nudgev {

struct TensorInfo {
  std::vector<int64_t> shape;
  ONNXTensorElementDataType data_type;
  bool is_initializer;
  const ONNX_NAMESPACE::TensorProto* initializer_tensor;
};


class NudgevModelWrapper {
 public:
  NudgevModelWrapper(const GraphViewer& graph_viewer,
                     const logging::Logger& logger,
                     const std::unordered_map<std::string, size_t>& input_index_map,
                     const std::unordered_map<std::string, size_t>& output_index_map,
                     const std::unordered_set<std::string>& initializer_lookup)
      : graph_viewer_(graph_viewer),
        logger_(logger),
        input_index_map_(input_index_map),
        output_index_map_(output_index_map),
        initializer_lookup_(initializer_lookup) {}

  Status GetTensorInfo(const NodeUnitIODef& input, TensorInfo& tensor_info) const;

  const InitializedTensorSet& GetInitializerTensors() const {
    return graph_viewer_.GetAllInitializedTensors();
  }

  bool IsInitializerInput(const std::string& input_name) const {
    return initializer_lookup_.find(input_name) != initializer_lookup_.end();
  }

  bool IsGraphOutput(const std::string& tensor_name) const {
    return output_index_map_.find(tensor_name) != output_index_map_.end();
  }

  bool IsGraphInput(const std::string& tensor_name) const {
    return input_index_map_.find(tensor_name) != input_index_map_.end();
  }

  Status UnpackInitializerData(const ONNX_NAMESPACE::TensorProto& initializer,
                              std::vector<uint8_t>& unpacked_tensor) const;

  Status GetQDQParams(const NodeUnitIODef& tensor,
                    std::vector<float>& scales,
                    std::vector<int8_t>& zero_points) const;

 private:
  const GraphViewer& graph_viewer_;
  const logging::Logger& logger_;
  const std::unordered_map<std::string, size_t>& input_index_map_;
  const std::unordered_map<std::string, size_t>& output_index_map_;
  const std::unordered_set<std::string>& initializer_lookup_;
};

}  // namespace nudgev
}  // namespace onnxruntime
