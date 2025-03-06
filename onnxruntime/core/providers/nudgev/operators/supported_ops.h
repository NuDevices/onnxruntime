#pragma once

#include <unordered_set>
#include <string>

namespace onnxruntime {
namespace nudgev {

const std::unordered_set<std::string> kSupportedQuantizedOps = {
    "Conv",
    "Gemm",
    "Add",
    "MaxPool"
    // Add other supported quantized operators here
    // For example, "Add", "MatMul", "MaxPool", "AveragePool", etc.
};

inline bool IsQuantizedOperatorSupported(const std::string& op_type) {
  return kSupportedQuantizedOps.find(op_type) != kSupportedQuantizedOps.end();
}

const std::unordered_set<std::string> kSkipOperatorBeforeQuantized = {
    "DequantizeLinear",
};

inline bool ShouldSkipBeforeQuantized(const std::string& op_type) {
  return kSkipOperatorBeforeQuantized.find(op_type) != kSkipOperatorBeforeQuantized.end();
}

}  // namespace nudgev
}  // namespace onnxruntime