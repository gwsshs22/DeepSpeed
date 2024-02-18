// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include "ds_kernel_utils.h"
#include "ragged_dtypes.h"

template <typename T>
void launch_moe_scatter(T* moe_input,
                        int64_t* expert_count_cumsums,
                        int32_t* mapped_slots,
                        const T* activations,
                        const int32_t* expert_counts,
                        const int32_t* assignments,
                        const int32_t* offsets,
                        const int32_t n_channels,
                        const int32_t n_tokens,
                        const int32_t n_experts,
                        const int32_t n_top_k,
                        cudaStream_t stream);

void launch_moe_build_local_permute_mapping(
    int32_t* local_assignments,
    int32_t* local_offsets,
    const int64_t* local_expert_cumsum,
    const int64_t* local_per_expert_cumsum,
    const int32_t local_expert_counts_max,
    const int32_t ep_size,
    const int32_t n_local_experts,
    cudaStream_t stream);
