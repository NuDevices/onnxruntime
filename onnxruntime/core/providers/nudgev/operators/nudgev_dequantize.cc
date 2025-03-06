#include "core/providers/nudgev/operators/nudgev_dequantize.h"
#include "core/framework/op_kernel_context_internal.h"
#include <cstring>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include "core/platform/threadpool.h"
#include <immintrin.h>

namespace onnxruntime {
namespace nudgev {

void nudgev_dequantize_linear(
    const int8_t* input,
    float* output,
    int64_t total_elements,
    float scale,
    int8_t zero_point,
    onnxruntime::concurrency::ThreadPool* tp) {
  constexpr int vector_size = 16;  // Process 16 elements per main loop iteration
  const float scaled_zero_point = -zero_point * scale;
  const __m256 scale_vec = _mm256_set1_ps(scale);
  const __m256 scaled_zp_vec = _mm256_set1_ps(scaled_zero_point);

  auto process_chunk = [&](int64_t start_idx, int64_t end_idx) {
    int64_t i = start_idx;

    // Process 16 elements per iteration
    for (; i + vector_size <= end_idx; i += vector_size) {
      // Load 16 int8 elements
      const __m128i input16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));

      // Process first 8 elements
      __m256i input_i32 = _mm256_cvtepi8_epi32(input16);
      __m256 input_f = _mm256_cvtepi32_ps(input_i32);
      __m256 result = _mm256_fmadd_ps(input_f, scale_vec, scaled_zp_vec);
      _mm256_storeu_ps(output + i, result);

      // Process next 8 elements
      const __m128i high64 = _mm_srli_si128(input16, 8);
      input_i32 = _mm256_cvtepi8_epi32(high64);
      input_f = _mm256_cvtepi32_ps(input_i32);
      result = _mm256_fmadd_ps(input_f, scale_vec, scaled_zp_vec);
      _mm256_storeu_ps(output + i + 8, result);
    }

    // Process remaining elements in groups of 8
    for (; i + 8 <= end_idx; i += 8) {
      const __m128i input8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input + i));
      const __m256i input_i32 = _mm256_cvtepi8_epi32(input8);
      const __m256 input_f = _mm256_cvtepi32_ps(input_i32);
      const __m256 result = _mm256_fmadd_ps(input_f, scale_vec, scaled_zp_vec);
      _mm256_storeu_ps(output + i, result);
    }

    // Handle remaining elements
    for (; i < end_idx; ++i) {
      output[i] = input[i] * scale + scaled_zero_point;
    }
  };

  constexpr int64_t chunk_size = 1024;  // Multiple of 16 for alignment
  if (tp != nullptr) {
    const int64_t num_chunks = (total_elements + chunk_size - 1) / chunk_size;
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, num_chunks,
        [&](int64_t chunk_idx) {
          const int64_t start = chunk_idx * chunk_size;
          const int64_t end = std::min(start + chunk_size, total_elements);
          process_chunk(start, end);
        });
  } else {
    process_chunk(0, total_elements);
  }
}

void nudgev_dequantize_linear_int32(
    const int32_t* input,
    float* output,
    int64_t total_elements,
    float scale,
    int32_t zero_point,
    onnxruntime::concurrency::ThreadPool* tp) {
  constexpr int vector_size = 8;
  const float scaled_zero_point = -zero_point * scale;
  const __m256 scale_vec = _mm256_set1_ps(scale);
  const __m256 scaled_zp_vec = _mm256_set1_ps(scaled_zero_point);

  auto process_chunk = [&](int64_t start_idx, int64_t end_idx) {
    int64_t i = start_idx;

    for (; i + vector_size <= end_idx; i += vector_size) {
      const __m256i input_i32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + i));
      const __m256 input_f = _mm256_cvtepi32_ps(input_i32);
      const __m256 result = _mm256_fmadd_ps(input_f, scale_vec, scaled_zp_vec);
      _mm256_storeu_ps(output + i, result);
    }

    // Process remaining elements
    for (; i < end_idx; ++i) {
      output[i] = input[i] * scale + scaled_zero_point;
    }
  };

  constexpr int64_t chunk_size = 1024;  // Multiple of 8 for alignment
  if (tp != nullptr) {
    const int64_t num_chunks = (total_elements + chunk_size - 1) / chunk_size;
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, num_chunks,
        [&](int64_t chunk_idx) {
          const int64_t start = chunk_idx * chunk_size;
          const int64_t end = std::min(start + chunk_size, total_elements);
          process_chunk(start, end);
        });
  } else {
    process_chunk(0, total_elements);
  }
}

Status ComputeNudgeVDequantizeLinear(DequantizeLinearParams* dequantize_params, OrtKernelContext* context) {
  auto start = std::chrono::high_resolution_clock::now();

  if (!dequantize_params) {
    return Status(common::ONNXRUNTIME, common::FAIL, "dequantize_params is null");
  }

  auto* ctx_internal = reinterpret_cast<OpKernelContextInternal*>(context);
  concurrency::ThreadPool* tp = ctx_internal->GetOperatorThreadPool();

  const Tensor* input = ctx_internal->Input<Tensor>(0);
  if (!input) {
    return Status(common::ONNXRUNTIME, common::FAIL, "input tensor is null");
  }

  const auto& input_shape = input->Shape();
  auto data_type = input->DataType();

  Tensor* output = ctx_internal->Output(0, input_shape);
  if (!output) {
    return Status(common::ONNXRUNTIME, common::FAIL, "failed to create output tensor");
  }

  auto* output_data = output->MutableData<float>();
  if (!output_data) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Output data is null");
  }

  const int64_t total_elements = input->Shape().Size();

  if (data_type == DataTypeImpl::GetType<int8_t>()) {
    const auto* input_data = input->Data<int8_t>();
    if (!input_data) {
      return Status(common::ONNXRUNTIME, common::FAIL, "Input data is null");
    }

    nudgev_dequantize_linear(
        input_data,
        output_data,
        total_elements,
        dequantize_params->scale,
        dequantize_params->zero_point,
        tp);
  } else if (data_type == DataTypeImpl::GetType<int32_t>()) {
    const auto* input_data = input->Data<int32_t>();
    if (!input_data) {
      return Status(common::ONNXRUNTIME, common::FAIL, "Input data is null");
    }

    int32_t zero_point_i32 = static_cast<int32_t>(dequantize_params->zero_point);

    nudgev_dequantize_linear_int32(
        input_data,
        output_data,
        total_elements,
        dequantize_params->scale,
        zero_point_i32,
        tp);
  } else {
    return Status(common::ONNXRUNTIME, common::FAIL,
                  "Unsupported input data type: " + std::string(DataTypeImpl::ToString(data_type)));
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "DequantizeLinear Op - Total execution time: " << duration.count() << " microseconds" << std::endl;

  return Status::OK();
}

}  // namespace nudgev
}  // namespace onnxruntime