import deepspeed.comm as dist
from deepspeed.inference.v2.checkpoint.in_memory_engine import InMemoryModelEngine
from deepspeed.inference.v2.model_implementations import MixtralPolicy
from deepspeed.inference.v2.engine_v2 import InferenceEngineV2
from deepspeed.inference.v2.config_v2 import RaggedInferenceEngineConfig
from deepspeed.inference.v2.inference_utils import enable_simulated_gating, disable_simulated_gating
from deepspeed.inference.v2.kernels.ragged_ops.top_k_gating.expert_probs import clear_expert_probs
from unit.common import DistributedTest
from unit.inference.v2.inference_test_utils import allclose

import torch
import pytest
import transformers
from transformers import MixtralForCausalLM, MixtralConfig

import os
import random

class TestMoeExpertParallelism(DistributedTest):
    world_size = 4

    def _build_mixtral_model(
        self,
        n_top_k,
        n_experts,
        num_hidden_layers=2,
        model_dim=512,
        intermediate_size=1792,
        vocab_size=32000
    ):
        model_config = MixtralConfig(
            num_local_experts=n_experts,
            num_experts_per_tok=n_top_k,
            num_hidden_layers=num_hidden_layers,
            model_dim=model_dim,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            torch_dtype="bfloat16"
        )
        model = MixtralForCausalLM(model_config)
        return model, model_config

    def _build_engine(self, model, model_config, max_seq_len, enable_ep, enable_simulated_gating):
        disable_simulated_gating()
        clear_expert_probs()

        raw_engine_config = {
            "state_manager": {
                "memory_config": {
                    "mode": "allocate",
                    "size": 16 * max_seq_len * self.world_size
                }
            },
            "tensor_parallel": {
                "tp_size": 1
            },
            "simulated_gating": enable_simulated_gating
        }

        if enable_ep:
            raw_engine_config["expert_parallel"] = {
                "enabled": True,
                "replica_num": self.world_size
            }

        engine_config = RaggedInferenceEngineConfig(**raw_engine_config)
        checkpoint_engine = InMemoryModelEngine(model)
        policy = MixtralPolicy(model_config, checkpoint_engine=checkpoint_engine)
        return InferenceEngineV2(policy, engine_config)

    def _make_input(self, max_seq_len, vocab_size, input_gen_random_seed=None):
        if input_gen_random_seed:
            random.seed(input_gen_random_seed)
            torch.manual_seed(input_gen_random_seed)
        seq_len = random.randint(1, max_seq_len)
        return torch.randint(0, vocab_size, (seq_len,))

    def _assert_output(self, model, model_config, logits, input_tokens, max_seq_len, n_experts, enable_simulated_gating):
        local_rank = int(os.environ.get("LOCAL_RANK"))
        device = logits.device

        seq_len = input_tokens.shape[0]
        seq_lens = torch.empty(self.world_size, dtype=torch.int32, device=device)
        dist.all_gather(list(seq_lens.split([1] * self.world_size)), torch.tensor(seq_len, dtype=torch.int32, device=device))
        seq_lens = seq_lens.cpu().tolist()

        if local_rank == 0:
            print(f"seq_lens={seq_lens}")

        batched_input_tokens = torch.zeros((self.world_size, max_seq_len), device=device, dtype=torch.int32)
        batched_input_tokens[local_rank][:seq_len] = input_tokens.to(device)
        dist.all_reduce(batched_input_tokens)

        if local_rank == 0:
            # Build a new engine that serves the model without any parallelisms.
            disable_simulated_gating()
            engine = self._build_engine(model, model_config, max_seq_len, enable_ep=False, enable_simulated_gating=enable_simulated_gating)
            input_tokens_list = []
            for i in range(self.world_size):
                input_tokens_list.append(batched_input_tokens[i][:seq_lens[i]].to("cpu"))
            ret_logits = engine.put(list(range(self.world_size)), input_tokens_list, do_checks=False)
        else:
            ret_logits = torch.zeros((self.world_size,) +  logits.shape[1:], dtype=logits.dtype, device=device)

        print(f"[local_rank={local_rank}] ret_logits.shape={ret_logits.shape}")
        dist.all_reduce(ret_logits)
        expected_logits = ret_logits[local_rank].reshape((1,) + logits.shape[1:])

        expected_logits = torch.softmax(expected_logits, -1)
        logits = torch.softmax(logits, -1)

        if input_tokens.shape[0] > 0:
            assert allclose(expected_logits, logits, tolerances=(0, 5e-3))

    def _test_mixtral_model_moe_ep(self,
                                   n_top_k,
                                   n_experts,
                                   max_seq_len,
                                   num_layers=2,
                                   vocab_size=32000,
                                   input_gen_random_seed=None,
                                   test_empty_run=False,
                                   enable_simulated_gating=False):

        local_rank = int(os.environ.get("LOCAL_RANK"))
        if test_empty_run and local_rank > 0:
            input_tokens = torch.tensor([], dtype=torch.int32)
        else:
            input_tokens = self._make_input(max_seq_len, vocab_size, input_gen_random_seed)

        torch.manual_seed(5000)
        model, model_config = self._build_mixtral_model(n_experts=n_experts, vocab_size=vocab_size, n_top_k=n_top_k, num_hidden_layers=num_layers)

        engine = self._build_engine(model, model_config, max_seq_len, enable_ep=True, enable_simulated_gating=enable_simulated_gating)
        if input_tokens.shape[0] != 0:
            logits = engine.put([0], [input_tokens], do_checks=False)
        else:
            engine.empty_run()
            logits = torch.empty(0, vocab_size, device=f'cuda:{local_rank}')
        self._assert_output(model, model_config, logits, input_tokens, max_seq_len, n_experts, enable_simulated_gating)

    @pytest.mark.disag_moe
    @pytest.mark.parametrize("n_top_k", [1, 2])
    @pytest.mark.parametrize("n_experts", [2, 4, 8, 16])
    @pytest.mark.parametrize("max_seq_len", [4, 256])
    def test_mixtral_model_moe_ep(self, n_top_k, n_experts, max_seq_len):
        self._test_mixtral_model_moe_ep(n_top_k, n_experts, max_seq_len)

    @pytest.mark.disag_moe
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("n_experts,max_seq_len", [(16, 2)])
    def test_mixtral_model_moe_ep_zero_recv_tokens(self, n_experts, max_seq_len):
        input_gen_random_seed = 3000 + int(os.environ.get("LOCAL_RANK"))
        self._test_mixtral_model_moe_ep(
            n_top_k=1,
            n_experts=n_experts,
            max_seq_len=max_seq_len,
            input_gen_random_seed=input_gen_random_seed
        )

    @pytest.mark.disag_moe
    def test_mixtral_model_moe_ep_empty_run(self):
        self._test_mixtral_model_moe_ep(2, 8, 256, test_empty_run=True)

    @pytest.mark.disag_moe
    @pytest.mark.parametrize("n_top_k", [1, 2])
    @pytest.mark.parametrize("n_experts", [16])
    @pytest.mark.parametrize("max_seq_len", [16, 128])
    def test_mixtral_model_moe_ep(self, n_top_k, n_experts, max_seq_len):
        self._test_mixtral_model_moe_ep(n_top_k, n_experts, max_seq_len, num_layers=1, enable_simulated_gating=True)
