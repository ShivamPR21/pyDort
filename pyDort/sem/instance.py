from typing import Any, Dict, Optional

import numpy as np

from .evolution import StateEvolution


class InstanceSEM(StateEvolution):

    def __init__(self,
                 state_dim: int = 3,
                 obs_dim: int = 3,
                 dt: float = None,
                 kwargs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(state_dim, obs_dim, dt)

        self.hx, self.fx = kwargs['hx'], kwargs['fx']
        self.x = kwargs["state0"]

    def propagate(self, **kwargs) -> None:
        dt = self.dt
        if 'dt' in kwargs:
            dt = kwargs['dt']
        assert(dt is not None)

        self.x = self.fx(self.x, dt)
        super().propagate()

    def update(self, observation: np.ndarray) -> None:
        self.x = self.hx(observation)
        super().update()

    @property
    def state(self) -> Any:
        return self.x

    @property
    def state_covar(self) -> Any:
        return None

    @property
    def obs_covar(self) -> Any:
        return None

    @property
    def process_noise(self) -> Any:
        return None
