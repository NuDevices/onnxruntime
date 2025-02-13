#pragma once

#include <unordered_set>
#include <string>

namespace onnxruntime {
namespace nudgev {

static const std::unordered_set<std::string> supported_ops = {
    "Conv",
    "Add",
    "MaxPool",
    "Gemm"};

inline bool IsNodeSupported(const std::string& op_type) {
  return supported_ops.find(op_type) != supported_ops.end();
}

}  // namespace nudgev
}  // namespace onnxruntime