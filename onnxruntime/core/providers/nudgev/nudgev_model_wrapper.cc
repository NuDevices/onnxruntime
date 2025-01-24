#include "core/providers/nudgev/nudgev_model_wrapper.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace nudgev {

Status NudgevModelWrapper::GetTensorInfo(const NodeUnitIODef& input, TensorInfo& tensor_info) const {
  const std::string& name = input.node_arg.Name();

  const auto* shape_proto = input.node_arg.Shape();
  ORT_RETURN_IF_NOT(shape_proto != nullptr, "Shape information missing for ", name);

  tensor_info.shape.clear();
  for (const auto& dim : shape_proto->dim()) {
    ORT_RETURN_IF_NOT(dim.has_dim_value(), "Dynamic shape is not supported yet for ", name);
    tensor_info.shape.push_back(dim.dim_value());
  }

  const auto* type_proto = input.node_arg.TypeAsProto();
  ORT_RETURN_IF_NOT(type_proto != nullptr && type_proto->has_tensor_type(),
                    "Type information missing for ", name);
  tensor_info.data_type = static_cast<ONNXTensorElementDataType>(type_proto->tensor_type().elem_type());

  tensor_info.is_initializer = IsInitializerInput(name);
  if (tensor_info.is_initializer) {
    tensor_info.initializer_tensor = GetInitializerTensors().at(name);
  }

  return Status::OK();
}

Status NudgevModelWrapper::UnpackInitializerData(const ONNX_NAMESPACE::TensorProto& initializer,
                                                std::vector<uint8_t>& unpacked_tensor) const {
  if (initializer.data_location() == onnx::TensorProto_DataLocation_EXTERNAL) {
    ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(initializer,
                                                                  graph_viewer_.ModelPath(),
                                                                  unpacked_tensor));
  } else {
    ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(initializer,
                                                                  unpacked_tensor));
  }
  return Status::OK();
}

Status NudgevModelWrapper::GetQDQParams(const NodeUnitIODef& tensor,
                                       std::vector<float>& scales,
                                       std::vector<int8_t>& zero_points) const {
  if (!tensor.quant_param) {
    return Status::OK();
  }

  const auto& scale_name = tensor.quant_param->scale.Name();
  const auto& zp_name = tensor.quant_param->zero_point->Name();


  const auto& initializers = GetInitializerTensors();

  auto scale_it = initializers.find(scale_name);
  ORT_RETURN_IF(scale_it == initializers.end(), "Scale initializer not found for ", scale_name);

  std::vector<uint8_t> scale_data;
  ORT_RETURN_IF_ERROR(UnpackInitializerData(*scale_it->second, scale_data));
  const float* scale_ptr = reinterpret_cast<const float*>(scale_data.data());
  size_t scale_count = scale_data.size() / sizeof(float);
  scales.assign(scale_ptr, scale_ptr + scale_count);

  auto zp_it = initializers.find(zp_name);
  ORT_RETURN_IF(zp_it == initializers.end(), "Zero point initializer not found for ", zp_name);

  std::vector<uint8_t> zp_data;
  ORT_RETURN_IF_ERROR(UnpackInitializerData(*zp_it->second, zp_data));
  const int8_t* zp_ptr = reinterpret_cast<const int8_t*>(zp_data.data());
  size_t zp_count = zp_data.size() / sizeof(int8_t);
  zero_points.assign(zp_ptr, zp_ptr + zp_count);

  return Status::OK();
}

}  // namespace nudgev
}  // namespace onnxruntime
