import numpy as np
from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer
from typing import Dict, List, Optional, Union


class ConstantPolicy(BasePolicy):

    def __init__(self, action: float):
        super().__init__()

        self._action = action

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
    ) -> Batch:
        """Compute the action over the given batch data."""

        return Batch(act=[self._action])

    def process_fn(self, batch: Batch, buffer: ReplayBuffer,
                   indices: np.ndarray) -> Batch:
        """It is not required to implement, because there is no training in this policy."""
        pass

    def learn(self, batch: Batch, batch_size: int,
              repeat: int) -> Dict[str, List[float]]:
        """It is not required to implement, because there is no training in this policy."""
        return
