# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.pydantic_v1 import Field

from deepspeed.runtime.config_utils import DeepSpeedConfigModel
from .ragged import DSStateManagerConfig


class DeepSpeedTPConfig(DeepSpeedConfigModel):
    """ Configure tensor parallelism settings """

    tp_size: int = 1
    """ Number of devices to split the model across using tensor parallelism. """

class DeepSpeedEPConfig(DeepSpeedConfigModel):
    """ Configure expert parallelism settings """

    enabled: bool = False

    replica_num: int = 1
    """
    Number of model replicas. Each replics will serve (NUM_EXPERTS // replica_num) experts.
    If NUM_EXPERTS < replica_num, we apply tensor parallelism on top of EP.
    """

class RaggedInferenceEngineConfig(DeepSpeedConfigModel):
    """ Sets parameters for DeepSpeed Inference Engine. """

    tensor_parallel: DeepSpeedTPConfig = Field({}, alias="tp")
    """
    Configuration for tensor parallelism used to split the model across several
    GPUs. Expects a dictionary containing values for :any:`DeepSpeedTPConfig`.
    """

    expert_parallel: DeepSpeedEPConfig = Field({}, alias="ep")

    state_manager: DSStateManagerConfig = Field({}, alias="manager")
    """
    Configuration for managing persistent state
    """

    simulated_gating: bool = False

    simulated_gating_temperature: float = 1.0

    trace_enabled: bool = False
