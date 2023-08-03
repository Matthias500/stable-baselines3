import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None
from common.buffers import ReplayBuffer


class TaskBuffer(ABC):

    def __init__(self,
                 nr_tasks: int = 1,
                 total_buffer_size: int,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 device: Union[th.device, str] = "cpu",
                 n_envs: int = 1,
                 optimize_memory_usage: bool = False,
                 handle_timeout_termination: bool = True,
    ):
        super(TaskBuffer, self).__init__()
        self.total_buffer_size = total_buffer_size
        self.buffer_size = total_buffer_size // nr_tasks

        self.buffers = [ReplayBuffer(
                buffer_size=self.buffer_size,
                observation_space=observation_space,
                action_space=action_space,
                device=device,
                n_envs=n_envs,
                optimize_memory_usage=optimize_memory_usage,
            ) for i in range(nr_tasks)]


