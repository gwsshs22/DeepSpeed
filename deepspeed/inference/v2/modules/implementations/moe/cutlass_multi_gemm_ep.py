# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any, Dict, Optional, Tuple

import torch

from deepspeed.accelerator import get_accelerator
import deepspeed.comm as dist
from ....allocator import empty_from
from ....inference_utils import ActivationType, is_gated, collect_expert_dist
from ....kernels.core_ops import BlasLibLinear, CUDAGatedActivation
from ....kernels.ragged_ops import (
    MoESummarizeRecvTokenStat,
    MoEBuildLocalPermuteMapping,
    MoEGather,
    MoEScatter,
    RaggedTopKGating,
)
from ....ragged import RaggedBatchWrapper

from ...interfaces import DSMoEBase, DSMoERegistry
from ...configs import DSMoEConfig
from ....kernels.cutlass_ops import MoEGEMM
from ....inference_parameter import InferenceParameter
from ....tracer import record


@DSMoERegistry.register_module
class DSMultiGemmMoEEp(DSMoEBase):
    """
    MoE implementation based on the CUTLASS multi-GEMM with expert parallelism.
    """

    @staticmethod
    def name():
        return 'cutlass_multi_gemm_moe_ep'

    @staticmethod
    def supports_config(config: DSMoEConfig) -> bool:
        if config.input_dtype != config.output_dtype:
            return False

        if config.input_dtype != torch.float16 and config.input_dtype != torch.bfloat16:
            return False

        if config.top_k != 1 and config.top_k != 2:
            return False

        return True

    def __init__(self, config: DSMoEConfig, implementation_config: Dict[str, Any]) -> None:
        super().__init__(config, implementation_config)

        # Convenience variables for frequently accessed items.
        self.max_tokens = self._config.max_tokens
        self.ep_size = self._config.ep_size
        self.n_experts = self._config.n_experts
        self.n_local_experts = max(1, self.n_experts // self.ep_size)
        self.expert_tp_degree = max(1, self.ep_size // self.n_experts)
        self.n_top_k = self._config.top_k
        self.intermediate_dim = self._config.intermediate_features
        assert not collect_expert_dist(), "Use tensor parallelism to collect expert assignments."

        moe_op_act_fn = ActivationType.IDENTITY if is_gated(self._config.activation) else self._config.activation

        self._mlp_1 = MoEGEMM(fp_dtype=implementation_config['weight_dtype'], act_fn=moe_op_act_fn)
        self._mlp_2 = MoEGEMM(fp_dtype=implementation_config['weight_dtype'], act_fn=ActivationType.IDENTITY)

        if is_gated(self._config.activation):
            self._activation = CUDAGatedActivation(self._config.model_dim, self._config.input_dtype,
                                                   self._config.activation)
        else:
            self._activation = None

        self._gate_proj = BlasLibLinear(self._config.input_dtype)

        self._top_k_gate = RaggedTopKGating(config.input_dtype, num_layers=config.num_layers, n_experts=self.n_experts, n_top_k=self.n_top_k)
        self._moe_scatter = MoEScatter(config.input_dtype, config.model_dim)
        self._moe_summarize_recv_token_stat = MoESummarizeRecvTokenStat()
        self._moe_build_local_permute_mapping = MoEBuildLocalPermuteMapping(config.input_dtype, config.model_dim)
        self._moe_gather = MoEGather(config.input_dtype, config.model_dim, config.normalize_scores)

        self._create_buffers()

    def _create_buffers(self):

        # Gating buffers
        self._logits = torch.empty((self._config.max_tokens, self.n_experts),
                                   dtype=self._config.input_dtype,
                                   device=get_accelerator().current_device())
        self._expert_counts = torch.empty((self.n_experts, ),
                                          dtype=torch.int32,
                                          device=get_accelerator().current_device())
        self._scores = torch.empty((self._config.max_tokens, self.n_top_k),
                                   dtype=torch.float32,
                                   device=get_accelerator().current_device())
        self._assignments = torch.empty((self._config.max_tokens, self.n_top_k),
                                        dtype=torch.int32,
                                        device=get_accelerator().current_device())
        self._offsets = torch.empty((self._config.max_tokens, self.n_top_k),
                                    dtype=torch.int32,
                                    device=get_accelerator().current_device())

        # EP related buffers
        self._recv_expert_counts = torch.empty((self.ep_size, self.n_local_experts),
                                               dtype=torch.int32,
                                               device=get_accelerator().current_device())

        self._local_expert_cumsum = torch.empty((self.n_local_experts, ),
                                               dtype=torch.int64,
                                               device=get_accelerator().current_device())
        
        self._local_scores = torch.ones((self._config.max_tokens * self.n_top_k * self.ep_size, 1),
                                         dtype=torch.float32,
                                         device=get_accelerator().current_device())
        self._local_mapped_slots = torch.empty(
            (self._config.max_tokens * self.n_top_k * self.ep_size, 1),
            dtype=torch.int32,
            device=get_accelerator().current_device()
        )

        self._local_assignments = torch.empty((self._config.max_tokens * self.n_top_k * self.ep_size, 1),
                                              dtype=torch.int32,
                                              device=get_accelerator().current_device())
        self._local_offsets = torch.empty((self._config.max_tokens * self.n_top_k * self.ep_size, 1),
                                          dtype=torch.int32,
                                          device=get_accelerator().current_device())

        self._recv_expert_cumsum = torch.empty((self.ep_size, self.n_local_experts),
                                               dtype=torch.int64,
                                               device=get_accelerator().current_device())
        self._recv_expert_cumsum_cpu = torch.empty_like(self._recv_expert_cumsum, device="cpu")
        self._recv_per_expert_cumsum = torch.empty((self.ep_size, self.n_local_experts),
                                                   dtype=torch.int64,
                                                   device=get_accelerator().current_device())
        self._recv_per_expert_cumsum_cpu = torch.empty_like(self._recv_per_expert_cumsum, device="cpu")
        self._local_expert_counts = torch.empty((self.n_local_experts,),
                                                dtype=torch.int32,
                                                device=get_accelerator().current_device())
        self._local_expert_counts_cpu = torch.empty_like(self._local_expert_counts, device="cpu")

        # Scatter buffers
        self._moe_input = torch.empty((self._config.max_tokens * self.n_top_k, self._config.model_dim),
                                      dtype=self._config.input_dtype,
                                      device=get_accelerator().current_device())
        self._expert_cumsum = torch.empty((self._config.n_experts, ),
                                          dtype=torch.int64,
                                          device=get_accelerator().current_device())
        self._mapped_slots = torch.empty((self._config.max_tokens, self.n_top_k),
                                         dtype=torch.int32,
                                         device=get_accelerator().current_device())

        self._shuffled_moe_input = torch.empty((self._config.max_tokens * self.n_top_k * self.ep_size, self._config.model_dim),
                                               dtype=self._config.output_dtype,
                                               device=get_accelerator().current_device())

        self._permuted_moe_input = torch.empty((self._config.max_tokens * self.n_top_k * self.ep_size, self._config.model_dim),
                                               dtype=self._config.output_dtype,
                                               device=get_accelerator().current_device())

        # GEMM Buffers
        self._intermediate = torch.empty((self._config.max_tokens * self.n_top_k * self.ep_size, self._config.intermediate_features),
                                         dtype=self._config.output_dtype,
                                         device=get_accelerator().current_device())
        if self._activation is not None:
            self._gated_intermediate = torch.empty(
                (self._config.max_tokens * self.n_top_k * self.ep_size, self._config.intermediate_features * 2),
                dtype=self._config.output_dtype,
                device=get_accelerator().current_device())

        self._output_unordered = torch.empty((self._config.max_tokens * self.n_top_k * self.ep_size, self._config.model_dim),
                                             dtype=self._config.output_dtype,
                                             device=get_accelerator().current_device())

        # Gather buffer
        self._output = torch.empty((self._config.max_tokens, self._config.model_dim),
                                   dtype=self._config.output_dtype,
                                   device=get_accelerator().current_device())

    def transform_gate_param(self, param: torch.Tensor) -> InferenceParameter:
        """
        Ensures gate param is going to match the activation data type.
        """
        param = param.to(self._config.input_dtype)
        return InferenceParameter.initialize(param)

    def transform_moe_mlp_1_param(self, param: torch.Tensor) -> InferenceParameter:
        """
        Converts param to same data type as input and output.

        Parameters:
            param (torch.Tensor): Weight or bias tensor.
        """
        param = param.to(self._config.input_dtype)

        if len(param.shape) == 3:
            param = param.permute(0, 2, 1).contiguous()
        return InferenceParameter.initialize(param)

    def transform_moe_mlp_2_param(self, param: torch.Tensor) -> InferenceParameter:
        """
        Converts param to same data type as input and output.

        Parameters:
            param (torch.Tensor): Weight or bias tensor.
        """
        param = param.to(self._config.input_dtype)

        if len(param.shape) == 3:
            param = param.permute(0, 2, 1).contiguous()
        return InferenceParameter.initialize(param)

    @property
    def output(self) -> torch.Tensor:
        return self._output

    def _gate(self, hidden_states: torch.Tensor, batch_metadata: RaggedBatchWrapper,
              gate_w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get views on the buffers for gating
        logits = empty_from(self._logits, (hidden_states.shape[0], self._logits.shape[-1]))
        scores = empty_from(self._scores, (hidden_states.shape[0], self.n_top_k))
        assignments = empty_from(self._assignments, (hidden_states.shape[0], self.n_top_k))
        offsets = empty_from(self._offsets, (hidden_states.shape[0], self.n_top_k))

        self._gate_proj(logits, hidden_states, gate_w)
        self._expert_counts.zero_()
        self._top_k_gate(self._expert_counts, scores, assignments, offsets, logits, batch_metadata)

        return self._expert_cumsum, scores, assignments, offsets

    def _run_mlp(self,
                 moe_input,
                 mlp_1_w,
                 mlp_2_w,
                 mlp_1_b,
                 mlp_2_b,
                 expert_cumsum):
        if moe_input.shape[0] == 0:
            return moe_input

        # Get views on the buffers for GEMM
        # intermediate = empty_from(self._intermediate,
        #                           (moe_input.shape[0], self._intermediate.shape[-1]))
        # output_unordered = empty_from(self._output_unordered,
        #                               (moe_input.shape[0], self._output_unordered.shape[-1]))
        num_tokens = moe_input.shape[0]
        intermediate = self._intermediate[:num_tokens]
        output_unordered = self._output_unordered[:num_tokens]

        if self._activation is not None:
            # gated_intermediate = empty_from(
            #     self._gated_intermediate, (moe_input.shape[0], self._gated_intermediate.shape[-1]))
            gated_intermediate = self._gated_intermediate[:num_tokens]
            self._mlp_1(
                gated_intermediate,
                moe_input,
                mlp_1_w,
                expert_cumsum,
                mlp_1_b,
            )
            self._activation(intermediate, gated_intermediate)
        else:
            self._mlp_1(
                intermediate,
                moe_input,
                mlp_1_w,
                expert_cumsum,
                mlp_1_b,
            )

        self._mlp_2(
            output_unordered,
            intermediate,
            mlp_2_w,
            expert_cumsum,
            mlp_2_b,
        )

        return output_unordered

    def forward(self,
                hidden_states: torch.Tensor,
                batch_metadata: RaggedBatchWrapper,
                gate_w: torch.Tensor,
                mlp_1_w: torch.Tensor,
                mlp_2_w: torch.Tensor,
                mlp_1_b: Optional[torch.Tensor] = None,
                mlp_2_b: Optional[torch.Tensor] = None,
                ep_group: Any = None) -> torch.Tensor:
        empty_run = hidden_states is None

        if not empty_run:    
            expert_cumsum, scores, assignments, offsets = self._gate(hidden_states, batch_metadata, gate_w)

            mapped_slots = empty_from(self._mapped_slots, (hidden_states.shape[0], self.n_top_k))
            moe_input = empty_from(self._moe_input, (hidden_states.shape[0] * self.n_top_k, self._moe_input.shape[-1]))
        else:
            self._expert_counts.zero_()
            moe_input = torch.empty(0, self._moe_input.shape[-1], device=self._moe_input.device, dtype=self._moe_input.dtype)

        # Implementation adopted from https://github.com/stanford-futuredata/megablocks/blob/main/megablocks/layers/moe.py.
        # 1. All-to-all recv/send token counts.
        send_expert_counts = self._expert_counts.reshape(-1, self.n_local_experts)
        if self.expert_tp_degree > 1:
            send_expert_counts = send_expert_counts.repeat(self.expert_tp_degree, 1)

        with record("moe_a2a_1"):
            a2a_handle_1 = dist.all_to_all_single(
                self._recv_expert_counts,
                send_expert_counts,
                async_op=True,
                group=ep_group
            )
            # 2. First scatter tokens before communicating tokens.
            if not empty_run:
                self._moe_scatter(moe_input, self._expert_cumsum, mapped_slots, hidden_states, self._expert_counts,
                                assignments, offsets)

            a2a_handle_1.wait()

        # Repeat tokens for expert tensor parallelism
        recv_expert_counts_per_rank, send_expert_counts_per_rank, recv_expert_counts_max, total_recv_tokens = self._moe_summarize_recv_token_stat(
            self._recv_expert_cumsum,
            self._recv_per_expert_cumsum,
            self._local_expert_counts,
            self._recv_expert_cumsum_cpu,
            self._recv_per_expert_cumsum_cpu,
            self._local_expert_counts_cpu,
            self._recv_expert_counts,
            send_expert_counts
        )

        repeated_moe_input = moe_input if self.expert_tp_degree == 1 else moe_input.repeat(self.expert_tp_degree, 1)
        shuffled_moe_input = self._shuffled_moe_input[:total_recv_tokens]

        with record("moe_a2a_2"):
            a2a_handle_2 = dist.all_to_all_single(
                shuffled_moe_input,
                repeated_moe_input,
                output_split_sizes=recv_expert_counts_per_rank,
                input_split_sizes=send_expert_counts_per_rank,
                async_op=True,
                group=ep_group
            )

            # 3. Prepare local permute. We need local permute as tokens from different ranks are grouped by experts in contiguous chunks.
            if total_recv_tokens > 0:
                local_assignments = self._local_assignments[:total_recv_tokens]
                local_offsets = self._local_offsets[:total_recv_tokens]

                self._moe_build_local_permute_mapping(
                    local_assignments,
                    local_offsets,
                    self._recv_expert_cumsum,
                    self._recv_per_expert_cumsum,
                    recv_expert_counts_max)

            a2a_handle_2.wait()

        with record("moe_ffn"):
            if total_recv_tokens > 0:
                    local_scores = self._local_scores[:total_recv_tokens]
                    local_mapped_slots = self._local_mapped_slots[:total_recv_tokens]

                    permuted_moe_input = self._permuted_moe_input[:total_recv_tokens]
                    # 4. Do local permute on shuffled_moe_input.
                    self._moe_scatter(permuted_moe_input, self._local_expert_cumsum, local_mapped_slots, shuffled_moe_input, self._local_expert_counts,
                                    local_assignments, local_offsets)

                    # 5. Compute expert layers with permuted input.
                    permuted_output_unordered = self._run_mlp(
                        permuted_moe_input,
                        mlp_1_w,
                        mlp_2_w,
                        mlp_1_b,
                        mlp_2_b,
                        self._local_expert_cumsum
                    )

                    # 6. Do local unpermute with permuted_output_unordered.
                    # We reuse shuffled_moe_input as an output buffer.
                    self._moe_gather(shuffled_moe_input, permuted_output_unordered, local_scores, local_mapped_slots, self._local_expert_counts)

        with record("moe_a2a_3"):
            # 7. a2a back
            dist.all_to_all_single(
                repeated_moe_input,
                shuffled_moe_input,
                output_split_sizes=send_expert_counts_per_rank,
                input_split_sizes=recv_expert_counts_per_rank,
                group=ep_group
            )

        if not empty_run:
            # 8. Sum repeated_moe_input for expert tensor parallelism.
            if self.expert_tp_degree > 1:
                repeated_moe_input = repeated_moe_input.reshape(self.expert_tp_degree, -1, repeated_moe_input.shape[-1])
                repeated_moe_input = repeated_moe_input.sum(0)

            # 9. Gather tokens to follow original orders.
            output = self._output[:hidden_states.shape[0]]
            self._moe_gather(output, repeated_moe_input, scores, mapped_slots, self._expert_counts)

            return output, None, None
        else:
            return
