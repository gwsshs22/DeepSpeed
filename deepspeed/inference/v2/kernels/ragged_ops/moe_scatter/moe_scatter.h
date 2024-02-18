// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include "moe_scatter.cuh"
#include "ragged_dtypes.h"

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
                 torch::Tensor& offsets);

void moe_build_local_permute_mapping(torch::Tensor& local_assignments,
                                     torch::Tensor& local_offsets,
                                     torch::Tensor& local_expert_cumsum,
                                     torch::Tensor& local_per_expert_cumsum,
                                     int local_expert_counts_max);
