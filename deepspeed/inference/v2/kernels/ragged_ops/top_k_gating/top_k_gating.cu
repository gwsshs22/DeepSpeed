// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "conversion_utils.h"
#include "memory_access_utils.h"
#include "reduction_utils.h"
#include "top_k_gating.cuh"
#include "top_k_utils.h"

#include <curand_kernel.h>

using ROp = reduce::ROpType;

template <typename T, int TOP_K>
__global__ void top_k_gating_kernel(int32_t* expert_counts,
                                    float* scores,
                                    int32_t* assignments,
                                    int32_t* offsets,
                                    const T* logits,
                                    const RaggedBatchDescriptor* batch_metadata,
                                    const int32_t n_experts)
{
    const int32_t token_idx = blockIdx.x;
    const int32_t expert_idx = threadIdx.x;
    const int32_t max_warps = 1024 / hw_warp_size;

    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // Padding tokens do not require
    if (token_idx >= batch_metadata->n_tokens) {
        if (threadIdx.x == 0) {
#pragma unroll
            for (int i = 0; i < TOP_K; i++) {
                assignments[token_idx * TOP_K + i] = gating::unassigned;
                offsets[token_idx * TOP_K + i] = gating::unassigned;
            }
        }
        return;
    }

    const T* token_logits = logits + token_idx * n_experts;

    float logit_val;
    if (expert_idx < n_experts) {
        logit_val = conversion::to<float>(token_logits[expert_idx]);
    } else {
        reduce::init<ROp::Max>(&logit_val);
    }
    float reduce_val = logit_val;

    int32_t local_assigned_experts[TOP_K];
    float local_assigned_logits[TOP_K];

    // Training code tends to use ``torch.argmax`` to select the expert, which
    // which has ties broken by the lower index. Since our fused comparison algorithm
    // breaks ties by the higher index (since it's the lower 32-bits of the 64-bit
    // comparison), we invert the expert index to break ties by the lower index.
    int32_t inverted_expert = n_experts - expert_idx - 1;

    // Find the top k logits
    for (int i = 0; i < TOP_K; ++i) {
        const reduce::IdxReduceResult res =
            reduce::idx_reduce<ROp::Max, max_warps>(tb, warp, reduce_val, inverted_expert);
        local_assigned_experts[i] = n_experts - res.idx - 1;
        local_assigned_logits[i] = res.val;

        // Set the max logit to -inf so that it is not selected again
        if (threadIdx.x == n_experts - res.idx - 1) { reduce::init<ROp::Max>(&reduce_val); }
    }

    const float max_logit = local_assigned_logits[0];
    float softmax_sum = __expf(logit_val - max_logit);
    reduce::block<ROp::Add>(tb, warp, softmax_sum);

    for (int i = 0; i < TOP_K; ++i) {
        const float softmax = __expf(local_assigned_logits[i] - max_logit) / softmax_sum;

        if (threadIdx.x == 0) {
            scores[token_idx * TOP_K + i] = softmax;
            assignments[token_idx * TOP_K + i] = local_assigned_experts[i];
            offsets[token_idx * TOP_K + i] =
                atomicAdd(expert_counts + local_assigned_experts[i], 1);
        }
    }
}

template <typename T>
void launch_top_k_gating(int32_t* expert_counts,
                         float* scores,
                         int32_t* assignments,
                         int32_t* offsets,
                         const T* logits,
                         const RaggedBatchDescriptor* batch_metadata,
                         const int32_t n_tokens,
                         const int32_t n_experts,
                         const int32_t n_top_k,
                         cudaStream_t stream)
{
    const dim3 grid(n_tokens);
    const dim3 block(((n_experts + hw_warp_size - 1) / hw_warp_size) * hw_warp_size);

    TOP_K_SWITCH(n_top_k, [&] {
        top_k_gating_kernel<T, CONST_TOP_K><<<grid, block, 0, stream>>>(
            expert_counts, scores, assignments, offsets, logits, batch_metadata, n_experts);
    });
}

#define INSTANTIATE_top_k_KERNEL(T)                                                   \
    template void launch_top_k_gating<T>(int32_t * expert_counts,                     \
                                         float* scores,                               \
                                         int32_t* assignments,                        \
                                         int32_t* offsets,                            \
                                         const T* logits,                             \
                                         const RaggedBatchDescriptor* batch_metadata, \
                                         const int32_t n_tokens,                      \
                                         const int32_t n_experts,                     \
                                         const int32_t n_top_k,                       \
                                         cudaStream_t stream);

INSTANTIATE_top_k_KERNEL(float) INSTANTIATE_top_k_KERNEL(__half)
#ifdef BF16_AVAILABLE
    INSTANTIATE_top_k_KERNEL(__nv_bfloat16)
#endif

template <typename T, int TOP_K>
__global__ void simulated_top_k_gating_kernel(int32_t* expert_counts,
                                              float* scores,
                                              int32_t* assignments,
                                              int32_t* offsets,
                                              const T* logits,
                                              const float* expert_probs,
                                              const int32_t n_tokens,
                                              const int32_t n_experts,
                                              const RaggedBatchDescriptor* batch_metadata)
{
    const int32_t token_idx = blockIdx.x * gating::threads_per_block + threadIdx.x;
    // Padding tokens do not require
    if (token_idx >= batch_metadata->n_tokens) {
#pragma unroll
        for (int i = 0; i < TOP_K; i++) {
            assignments[token_idx * TOP_K + i] = gating::unassigned;
            offsets[token_idx * TOP_K + i] = gating::unassigned;
        }
        return;
    }

    float seed_float = conversion::to<float>(*(logits + token_idx * n_experts));
    u_int32_t seed = __float_as_uint(seed_float);
    curandStatePhilox4_32_10_t state;
    curand_init(seed, 0, 0, &state);

    int32_t local_assigned_experts[TOP_K];
    float local_assigned_scores[TOP_K];

    float rnd = curand_uniform(&state);
    int top_1_expert_id;
    float top_1_expert_prob;

    for (top_1_expert_id = 0; top_1_expert_id < n_experts; top_1_expert_id++) {
      top_1_expert_prob = expert_probs[top_1_expert_id];
      rnd -= top_1_expert_prob;
      if (rnd <= 0.0) {
        break;
      }
    }

    local_assigned_experts[0] = top_1_expert_id;
    local_assigned_scores[0] = 0.7;

    // Currenlty we only consider TOP_K = 1 or 2.
    if (TOP_K == 2) {
      rnd = curand_uniform(&state) * (1.0 - expert_probs[n_experts + top_1_expert_id]);
      int top_2_expert_id;
      float top_2_expert_prob;

      for (top_2_expert_id = 0; top_2_expert_id < n_experts; top_2_expert_id++) {
        if (top_2_expert_id == top_1_expert_id) {
          continue;
        }

        top_2_expert_prob = expert_probs[n_experts + top_2_expert_id];
        rnd -= top_2_expert_prob;
        if (rnd <= 0.0) {
          break;
        }
      }

      local_assigned_experts[1] = top_2_expert_id;
      local_assigned_scores[1] = 0.2;
    }

    float score_sum = 0.0;
#pragma unroll
    for (int i = 0; i < TOP_K; ++i) {
      score_sum += local_assigned_scores[i];
    }

#pragma unroll
    for (int i = 0; i < TOP_K; ++i) {
        scores[token_idx * TOP_K + i] = local_assigned_scores[i] / score_sum;
        assignments[token_idx * TOP_K + i] = local_assigned_experts[i];
        offsets[token_idx * TOP_K + i] =
            atomicAdd(expert_counts + local_assigned_experts[i], 1);
    }
}

template <typename T>
void launch_simulated_top_k_gating(int32_t* expert_counts,
                                   float* scores,
                                   int32_t* assignments,
                                   int32_t* offsets,
                                   const T* logits,
                                   const float* expert_probs,
                                   const int32_t n_tokens,
                                   const int32_t n_experts,
                                   const int32_t n_top_k,
                                   const RaggedBatchDescriptor* batch_metadata,
                                   cudaStream_t stream)
{
  const int32_t num_blocks = (n_tokens + gating::threads_per_block - 1) / gating::threads_per_block;
  const dim3 grid(num_blocks);
  const dim3 block(gating::threads_per_block);

  TOP_K_SWITCH(n_top_k, [&] {
      simulated_top_k_gating_kernel<T, CONST_TOP_K><<<grid, block, 0, stream>>>(expert_counts,
                                                                                scores,
                                                                                assignments,
                                                                                offsets,
                                                                                logits,
                                                                                expert_probs,
                                                                                n_tokens,
                                                                                n_experts,
                                                                                batch_metadata);
  });
}

#define INSTANTIATE_simulated_top_k_KERNEL(T)                                                \
    template void launch_simulated_top_k_gating(int32_t* expert_counts,                      \
                                                float* scores,                               \
                                                int32_t* assignments,                        \
                                                int32_t* offsets,                            \
                                                const T* logits,                             \
                                                const float* expert_probs,                   \
                                                const int32_t n_tokens,                      \
                                                const int32_t n_experts,                     \
                                                const int32_t n_top_k,                       \
                                                const RaggedBatchDescriptor* batch_metadata, \
                                                cudaStream_t stream);

INSTANTIATE_simulated_top_k_KERNEL(float) INSTANTIATE_simulated_top_k_KERNEL(__half)
#ifdef BF16_AVAILABLE
    INSTANTIATE_simulated_top_k_KERNEL(__nv_bfloat16)
#endif
