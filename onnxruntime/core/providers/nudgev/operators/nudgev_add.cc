#include "core/providers/nudgev/operators/nudgev_add.h"
#include "core/framework/op_kernel_context_internal.h"
#include <cstring>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include "core/platform/threadpool.h"
#include <immintrin.h>
#include <functional>

namespace onnxruntime {
namespace nudgev {

inline void compute_broadcast_indices(
    const std::vector<int64_t>& shape1,
    const std::vector<int64_t>& shape2,
    const std::vector<int64_t>& output_shape,
    int64_t flat_idx,
    int64_t& idx1,
    int64_t& idx2) {
  std::vector<int64_t> output_indices(output_shape.size(), 0);
  int64_t remaining = flat_idx;

  for (int i = static_cast<int>(output_shape.size()) - 1; i >= 0; --i) {
    output_indices[i] = remaining % output_shape[i];
    remaining /= output_shape[i];
  }

  idx1 = 0;
  int64_t stride1 = 1;
  for (int i = static_cast<int>(shape1.size()) - 1; i >= 0; --i) {
    int out_dim = i + (output_shape.size() - shape1.size());
    if (out_dim >= 0) {
      int64_t dim_idx = (shape1[i] == 1) ? 0 : output_indices[out_dim];
      idx1 += dim_idx * stride1;
    }
    stride1 *= shape1[i];
  }

  idx2 = 0;
  int64_t stride2 = 1;
  for (int i = static_cast<int>(shape2.size()) - 1; i >= 0; --i) {
    int out_dim = i + (output_shape.size() - shape2.size());
    if (out_dim >= 0) {
      int64_t dim_idx = (shape2[i] == 1) ? 0 : output_indices[out_dim];
      idx2 += dim_idx * stride2;
    }
    stride2 *= shape2[i];
  }
}

void add_quantized_int8(
    const int8_t* input1_data,
    const int8_t* input2_data,
    int8_t* output_data,
    const std::vector<int64_t>& input1_shape,
    const std::vector<int64_t>& input2_shape,
    const std::vector<int64_t>& output_shape,
    int8_t input1_zp,
    int8_t input2_zp,
    int8_t output_zp,
    int32_t M1_fixed,
    int32_t M2_fixed,
    int64_t total_elements,
    bool fused_relu,
    onnxruntime::concurrency::ThreadPool* tp) {
  const bool no_broadcast = (input1_shape == output_shape && input2_shape == output_shape);

  int32_t zp_correction = (static_cast<int32_t>(input1_zp) * M1_fixed +
                           static_cast<int32_t>(input2_zp) * M2_fixed + 0x4000) >>
                          15;
  zp_correction -= static_cast<int32_t>(output_zp);

  auto process_chunk = [&](int64_t start_idx, int64_t end_idx) {
    if (no_broadcast) {
      for (int64_t i = start_idx; i < end_idx; ++i) {
        int32_t val1 = static_cast<int32_t>(input1_data[i]);
        int32_t val2 = static_cast<int32_t>(input2_data[i]);

        int32_t scaled_val1 = static_cast<int32_t>((static_cast<int64_t>(val1) * M1_fixed + 0x4000) >> 15);
        int32_t scaled_val2 = static_cast<int32_t>((static_cast<int64_t>(val2) * M2_fixed + 0x4000) >> 15);

        int32_t sum = scaled_val1 + scaled_val2 - zp_correction;
        if (fused_relu && sum < 0) {
          sum = 0;
        }

        sum = std::min(127, std::max(-128, sum));

        output_data[i] = static_cast<int8_t>(sum);
      }
    } else {
      for (int64_t i = start_idx; i < end_idx; ++i) {
        int64_t idx1, idx2;
        compute_broadcast_indices(input1_shape, input2_shape, output_shape, i, idx1, idx2);

        int32_t val1 = static_cast<int32_t>(input1_data[idx1]);
        int32_t val2 = static_cast<int32_t>(input2_data[idx2]);

        int32_t scaled_val1 = static_cast<int32_t>((static_cast<int64_t>(val1) * M1_fixed + 0x4000) >> 15);
        int32_t scaled_val2 = static_cast<int32_t>((static_cast<int64_t>(val2) * M2_fixed + 0x4000) >> 15);

        int32_t sum = scaled_val1 + scaled_val2 - zp_correction;

        if (fused_relu && sum < 0) {
          sum = 0;
        }

        sum = std::min(127, std::max(-128, sum));

        output_data[i] = static_cast<int8_t>(sum);
      }
    }
  };

  if (tp != nullptr && total_elements >= 4096) {
    const int64_t chunk_size = 1024;
    const int64_t num_chunks = (total_elements + chunk_size - 1) / chunk_size;

    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, num_chunks, [&](int64_t chunk_idx) {
          const int64_t start = chunk_idx * chunk_size;
          const int64_t end = std::min(start + chunk_size, total_elements);
          process_chunk(start, end);
        });
  } else {
    process_chunk(0, total_elements);
  }
}

std::vector<int64_t> compute_broadcast_output_shape(
    const std::vector<int64_t>& shape1,
    const std::vector<int64_t>& shape2) {
  const size_t rank1 = shape1.size();
  const size_t rank2 = shape2.size();
  const size_t max_rank = std::max(rank1, rank2);

  std::vector<int64_t> output_shape(max_rank);

  for (size_t i = 0; i < max_rank; ++i) {
    const int64_t dim1 = (i < rank1) ? shape1[rank1 - i - 1] : 1;
    const int64_t dim2 = (i < rank2) ? shape2[rank2 - i - 1] : 1;

    if (dim1 == 1 || dim2 == 1 || dim1 == dim2) {
      output_shape[max_rank - i - 1] = std::max(dim1, dim2);
    } else {
      throw std::runtime_error("Incompatible dimensions for broadcasting");
    }
  }

  return output_shape;
}

Status ComputeNudgeVAdd(AddParams* add_params, OrtKernelContext* context) {
  auto start = std::chrono::high_resolution_clock::now();

  if (!add_params) {
    return Status(common::ONNXRUNTIME, common::FAIL, "add_params is null");
  }

  auto* ctx_internal = reinterpret_cast<OpKernelContextInternal*>(context);
  concurrency::ThreadPool* tp = ctx_internal->GetOperatorThreadPool();

  const Tensor* input1 = ctx_internal->Input<Tensor>(0);
  const Tensor* input2 = ctx_internal->Input<Tensor>(3);

  if (!input1 || !input2) {
    return Status(common::ONNXRUNTIME, common::FAIL, "input tensor is null");
  }

  const auto& input1_shape_span = input1->Shape().GetDims();
  const auto& input2_shape_span = input2->Shape().GetDims();

  // Converti gli span in vettori
  std::vector<int64_t> input1_shape(input1_shape_span.begin(), input1_shape_span.end());
  std::vector<int64_t> input2_shape(input2_shape_span.begin(), input2_shape_span.end());

  // Calcola la forma di output con broadcasting
  std::vector<int64_t> output_shape;
  try {
    output_shape = compute_broadcast_output_shape(input1_shape, input2_shape);
  } catch (const std::runtime_error& e) {
    return Status(common::ONNXRUNTIME, common::FAIL,
                  "Failed to compute broadcast shape: " + std::string(e.what()));
  }

  TensorShape output_tensor_shape(output_shape);
  Tensor* Y = ctx_internal->Output(0, output_tensor_shape);

  if (!Y) {
    return Status(common::ONNXRUNTIME, common::FAIL, "failed to create output tensor");
  }

  if (Y->DataType() != DataTypeImpl::GetType<int8_t>()) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Output type must be int8");
  }

  const auto* input1_data = input1->Data<int8_t>();
  const auto* input2_data = input2->Data<int8_t>();
  auto* output_data = Y->MutableData<int8_t>();

  if (!input1_data || !input2_data || !output_data) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Input or output data is null");
  }

  const int64_t total_elements = output_tensor_shape.Size();

  add_quantized_int8(
      input1_data,
      input2_data,
      output_data,
      input1_shape,
      input2_shape,
      output_shape,
      add_params->input1_zp,
      add_params->input2_zp,
      add_params->output_zp,
      add_params->M1_fixed,
      add_params->M2_fixed,
      total_elements,
      add_params->fused_relu,
      tp);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Add Op - Total execution time: " << duration.count() << " microseconds" << std::endl;

  return Status::OK();
}

}  // namespace nudgev
}  // namespace onnxruntime