#include "core/providers/nudgev/operators/nudgev_gemm.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/providers/nudgev/nudgev_execution_provider.h"
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

void nudgev_gemm(
    const int8_t* input,
    const int8_t* weights,
    int8_t* output,
    int64_t M,
    int64_t K,
    int64_t N,
    const int32_t* bias,
    bool has_bias,
    int32_t M_fixed,
    int8_t input_zp,
    int8_t output_zp,
    bool fused_relu,
    onnxruntime::concurrency::ThreadPool* tp) {
  constexpr int vector_size = 16;

  std::vector<int32_t> zp_compensation(N);
  for (int n = 0; n < N; ++n) {
    int32_t weight_sum = 0;
    const int8_t* weight_row = weights + n * K;
    for (int k = 0; k < K; ++k) {
      weight_sum += static_cast<int32_t>(weight_row[k]);
    }
    zp_compensation[n] = weight_sum * static_cast<int32_t>(input_zp);
  }

  auto process_chunk = [&](int64_t start_idx, int64_t end_idx) {
    for (int64_t idx = start_idx; idx < end_idx; ++idx) {
      int64_t m = idx / N;
      int64_t n = idx % N;
      const int8_t* input_row = input + m * K;
      int8_t* output_ptr = output + m * N;
      const int8_t* weight_row = weights + n * K;

      __m256i acc_lo = _mm256_setzero_si256();
      __m256i acc_hi = _mm256_setzero_si256();

      int64_t k = 0;
      for (; k <= K - vector_size; k += vector_size) {
        __m128i inp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input_row + k));
        __m128i w = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weight_row + k));

        __m256i inp_epi16 = _mm256_cvtepi8_epi16(inp);
        __m256i w_epi16 = _mm256_cvtepi8_epi16(w);

        __m256i prod = _mm256_mullo_epi16(w_epi16, inp_epi16);

        __m128i prod_lo = _mm256_castsi256_si128(prod);
        __m128i prod_hi = _mm256_extracti128_si256(prod, 1);

        acc_lo = _mm256_add_epi32(acc_lo, _mm256_cvtepi16_epi32(prod_lo));
        acc_hi = _mm256_add_epi32(acc_hi, _mm256_cvtepi16_epi32(prod_hi));
      }

      alignas(32) int32_t acc_array[8];
      _mm256_store_si256(reinterpret_cast<__m256i*>(acc_array), acc_lo);

      int32_t acc = 0;
      for (int i = 0; i < 8; ++i) {
        acc += acc_array[i];
      }

      _mm256_store_si256(reinterpret_cast<__m256i*>(acc_array), acc_hi);
      for (int i = 0; i < 8; ++i) {
        acc += acc_array[i];
      }
      for (; k < K; ++k) {
        acc += static_cast<int32_t>(weight_row[k]) * static_cast<int32_t>(input_row[k]);
      }

      acc -= zp_compensation[n];
      if (has_bias) {
        acc += bias[n];
      }

      int64_t scaled_acc = static_cast<int64_t>(acc) * M_fixed;
      int32_t sum = static_cast<int32_t>((scaled_acc + 0x4000) >> 15);

      if (fused_relu) {
        sum = std::max(sum, 0);
      }

      sum += output_zp;
      sum = std::min(127, std::max(-128, sum));

      output_ptr[n] = static_cast<int8_t>(sum);
    }
  };

  const int64_t total_work = M * N;
  if (tp != nullptr) {
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, total_work,
        [&](int64_t work_idx) {
          process_chunk(work_idx, work_idx + 1);
        });
  } else {
    process_chunk(0, total_work);
  }
}

Status ComputeNudgeVGemm(GemmParams* gemm_params, OrtKernelContext* context) {
  auto start_gemm = std::chrono::high_resolution_clock::now();

  if (!gemm_params) {
    return Status(common::ONNXRUNTIME, common::FAIL, "gemm_params is null");
  }

  auto* ctx_internal = reinterpret_cast<OpKernelContextInternal*>(context);
  concurrency::ThreadPool* tp = ctx_internal->GetOperatorThreadPool();

  const Tensor* input = ctx_internal->Input<Tensor>(0);
  if (!input) {
    return Status(common::ONNXRUNTIME, common::FAIL, "input tensor is null");
  }

  const auto& input_shape = input->Shape();

  if (input_shape.NumDimensions() < 2) {
    return Status(common::ONNXRUNTIME, common::FAIL, "GEMM input A must have at least 2 dimensions");
  }

  const int64_t actual_batch_size = input_shape[0];
  if (gemm_params->dynamic_batch && actual_batch_size != gemm_params->batch_size) {
    gemm_params->batch_size = actual_batch_size;
  }

  const int64_t M = gemm_params->batch_size;
  const int64_t K = gemm_params->K;
  const int64_t N = gemm_params->N;

  if (gemm_params->transA) {
    if (input_shape[1] != M || input_shape[0] != K) {
      return Status(common::ONNXRUNTIME, common::FAIL,
                    "Input dimensions don't match expected dimensions for transposed A");
    }
  } else {
    if (input_shape[0] != M || input_shape[1] != K) {
      return Status(common::ONNXRUNTIME, common::FAIL,
                    "Input dimensions don't match expected dimensions");
    }
  }

  std::vector<int64_t> output_shape{M, N};
  Tensor* Y = ctx_internal->Output(0, output_shape);
  if (!Y) {
    return Status(common::ONNXRUNTIME, common::FAIL, "failed to create output tensor");
  }

  if (Y->DataType() != DataTypeImpl::GetType<int8_t>()) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Output type must be int8");
  }

  const auto* input_data = input->Data<int8_t>();
  auto* output_data = Y->MutableData<int8_t>();

  if (!gemm_params->weights_buffer.data()) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Weights buffer is null");
  }

  if (!input_data) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Input data is null");
  }

  if (!output_data) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Output data is null");
  }

  onnxruntime::nudgev::nudgev_gemm(
      input_data,
      gemm_params->weights_buffer.data(),
      output_data,
      M,
      K,
      N,
      gemm_params->bias_buffer.data(),
      !gemm_params->bias_buffer.empty(),
      gemm_params->M_fixed,
      gemm_params->input_zp,
      gemm_params->output_zp,
      false,
      tp);

  auto end_gemm = std::chrono::high_resolution_clock::now();
  auto duration_gemm = std::chrono::duration_cast<std::chrono::microseconds>(end_gemm - start_gemm);
  std::cout << "Gemm Op - Total execution time: " << duration_gemm.count() << " microseconds" << std::endl;

  return Status::OK();
}
}  // namespace nudgev
}  // namespace onnxruntime