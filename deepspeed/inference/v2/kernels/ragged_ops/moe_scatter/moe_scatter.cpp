// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "moe_scatter.h"
#include <c10/cuda/CUDAStream.h>

#define DISPATCH_MOE_SCATTER(T_TYPE, C_TYPE)                          \
    if (activations.options().dtype() == torch::T_TYPE) {             \
        launch_moe_scatter((C_TYPE*)moe_input.data_ptr(),             \
                           (int64_t*)expert_count_cumsums.data_ptr(), \
                           (int32_t*)mapped_slots.data_ptr(),         \
                           (const C_TYPE*)activations.data_ptr(),     \
                           (const int32_t*)expert_counts.data_ptr(),  \
                           (const int32_t*)assignments.data_ptr(),    \
                           (const int32_t*)offsets.data_ptr(),        \
                           n_channels,                                \
                           n_tokens,                                  \
                           n_experts,                                 \
                           n_top_k,                                   \
                           at::cuda::getCurrentCUDAStream());         \
        return;                                                       \
    }

/*
Performs a cumsum on the expert counts and copies the hidden states to the
appropriate spot to ensure that each experts inputs are contiguous.
*/
void moe_scatter(torch::Tensor& moe_input,
                 torch::Tensor& expert_count_cumsums,
                 torch::Tensor& mapped_slots,
                 torch::Tensor& activations,
                 torch::Tensor& expert_counts,
                 torch::Tensor& assignments,
                 torch::Tensor& offsets)
{
    const int32_t n_tokens = activations.size(0);
    const int32_t n_channels = activations.size(1);
    const int32_t n_top_k = assignments.size(1);

    // Should have a lot of matching buffer sizes here.
    TORCH_CHECK(n_tokens == assignments.size(0));
    TORCH_CHECK(n_tokens == offsets.size(0));
    TORCH_CHECK(n_channels == moe_input.size(1));

    TORCH_CHECK(n_top_k == offsets.size(1));
    TORCH_CHECK(n_top_k * n_tokens == moe_input.size(0));
    TORCH_CHECK(n_top_k == mapped_slots.size(1));

    const int32_t n_experts = expert_count_cumsums.size(0);

    TORCH_CHECK(moe_input.scalar_type() == activations.scalar_type());
    TORCH_CHECK(expert_count_cumsums.scalar_type() == torch::kInt64);
    TORCH_CHECK(mapped_slots.scalar_type() == torch::kInt32);
    TORCH_CHECK(expert_counts.scalar_type() == torch::kInt32);
    TORCH_CHECK(assignments.scalar_type() == torch::kInt32);
    TORCH_CHECK(offsets.scalar_type() == torch::kInt32);

    DISPATCH_MOE_SCATTER(kHalf, __half);

#ifdef BF16_AVAILABLE
    DISPATCH_MOE_SCATTER(kBFloat16, __nv_bfloat16);
#endif

    TORCH_CHECK(false, "Unsupported dtype for moe_scatter")
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>, int32_t, int64_t> moe_summarize_recv_token_stat(
    torch::Tensor& recv_expert_cumsum,
    torch::Tensor& recv_per_expert_cumsum,
    torch::Tensor& local_expert_counts,
    torch::Tensor& recv_expert_cumsum_cpu,
    torch::Tensor& recv_per_expert_cumsum_cpu,
    torch::Tensor& local_expert_counts_cpu,
    torch::Tensor& recv_expert_counts,
    torch::Tensor& send_expert_counts) {
  const int32_t ep_size = recv_expert_counts.size(0);
  const int32_t n_local_experts = recv_expert_counts.size(1);

  TORCH_CHECK(recv_expert_cumsum.size(0) == ep_size);
  TORCH_CHECK(recv_expert_cumsum.size(1) == n_local_experts);
  TORCH_CHECK(recv_per_expert_cumsum.size(0) == ep_size);
  TORCH_CHECK(recv_per_expert_cumsum.size(1) == n_local_experts);
  TORCH_CHECK(local_expert_counts.size(0) == n_local_experts);
  TORCH_CHECK(send_expert_counts.size(0) == ep_size);
  TORCH_CHECK(send_expert_counts.size(1) == n_local_experts);

  auto stream = at::cuda::getCurrentCUDAStream();
  torch::Tensor recv_expert_counts_cpu = recv_expert_counts.to(torch::kCPU, true, true);
  torch::Tensor send_expert_counts_cpu = send_expert_counts.to(torch::kCPU, true, true);

  std::vector<int64_t> recv_expert_counts_per_rank(ep_size);
  std::vector<int64_t> send_expert_counts_per_rank(ep_size);
  
  stream.synchronize();

  int32_t recv_expert_counts_max = 0;
  auto recv_expert_counts_cpu_ptr = recv_expert_counts_cpu.accessor<int32_t, 2>();
  auto recv_expert_cumsum_cpu_ptr = recv_expert_cumsum_cpu.accessor<int64_t, 2>();
  int64_t total_recv_tokens = 0;
  
  for (int i = 0; i < ep_size; i++) {
    int64_t sum = 0;
    for (int j = 0; j < n_local_experts; j++) {
      int tmp = recv_expert_counts_cpu_ptr[i][j];
      sum += tmp;
      total_recv_tokens += tmp;
      recv_expert_cumsum_cpu_ptr[i][j] = total_recv_tokens;

      if (tmp > recv_expert_counts_max) {
        recv_expert_counts_max = tmp;
      }
    }

    recv_expert_counts_per_rank[i] = sum;
  }
  recv_expert_cumsum.copy_(recv_expert_cumsum_cpu, /* non_blocking */ true);

  auto recv_per_expert_cumsum_cpu_ptr = recv_per_expert_cumsum_cpu.accessor<int64_t, 2>();
  auto local_expert_counts_cpu_ptr = local_expert_counts_cpu.accessor<int32_t, 1>();
  for (int j = 0; j < n_local_experts; j++) {
    int64_t sum = 0;
    for (int i = 0; i < ep_size; i++) {
      sum += recv_expert_counts_cpu_ptr[i][j];
      recv_per_expert_cumsum_cpu_ptr[i][j] = sum;
    }

    local_expert_counts_cpu_ptr[j] = sum;
  }

  recv_per_expert_cumsum.copy_(recv_per_expert_cumsum_cpu, /* non_blocking */  true);
  local_expert_counts.copy_(local_expert_counts_cpu, /* non_blocking */  true);

  auto send_expert_counts_cpu_ptr = send_expert_counts_cpu.accessor<int32_t, 2>();
  for (int i = 0; i < ep_size; i++) {
    int64_t sum = 0;
    for (int j = 0; j < n_local_experts; j++) {
      sum += send_expert_counts_cpu_ptr[i][j];
    }
    send_expert_counts_per_rank[i] = sum;
  }

  return std::make_tuple(recv_expert_counts_per_rank, send_expert_counts_per_rank, recv_expert_counts_max, total_recv_tokens);
}

void moe_build_local_permute_mapping(torch::Tensor& local_assignments,
                                     torch::Tensor& local_offsets,
                                     torch::Tensor& local_expert_cumsum,
                                     torch::Tensor& local_per_expert_cumsum,
                                     int32_t local_expert_counts_max) {
  
  const int32_t n_tokens = local_assignments.size(0);
  const int32_t ep_size = local_expert_cumsum.size(0);
  const int32_t n_local_experts = local_expert_cumsum.size(1);

  TORCH_CHECK(local_assignments.size(0) == n_tokens);
  TORCH_CHECK(ep_size == local_per_expert_cumsum.size(0));
  TORCH_CHECK(n_local_experts == local_per_expert_cumsum.size(1));

  TORCH_CHECK(local_assignments.scalar_type() == torch::kInt32);
  TORCH_CHECK(local_offsets.scalar_type() == torch::kInt32);
  TORCH_CHECK(local_expert_cumsum.scalar_type() == torch::kInt64);
  TORCH_CHECK(local_per_expert_cumsum.scalar_type() == torch::kInt64);

  launch_moe_build_local_permute_mapping(
    (int32_t*)local_assignments.data_ptr(),
    (int32_t*)local_offsets.data_ptr(),
    (const int64_t*)local_expert_cumsum.data_ptr(),
    (const int64_t*)local_per_expert_cumsum.data_ptr(),
    (const int32_t)local_expert_counts_max,
    ep_size,
    n_local_experts,
    at::cuda::getCurrentCUDAStream()
  );
}
