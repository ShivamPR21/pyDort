from typing import Any, Dict, Optional

import numpy as np
from filterpy import kalman

from .evolution import StateEvolution


class FilterPyUKF(StateEvolution):

    def __init__(self,
                 state_dim: int = 3,
                 obs_dim: int = 3,
                 dt: float = None,
                 kwargs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(state_dim, obs_dim, dt)

        points = kalman.MerweScaledSigmaPoints(self.state_dim, kwargs['alpha'], 2., -1)
        self.ukf = kalman.UnscentedKalmanFilter(
                            self.state_dim,
                            self.obs_dim,
                            self.dt,
                            kwargs['hx'],
                            kwargs['fx'],
                            points)
        self.ukf.x = kwargs["state0"]
        self.ukf.P = kwargs["P0"]
        self.ukf.Q = kwargs["Q"]
        self.ukf.R = kwargs["R"]

    def propagate(self, **kwargs) -> None:
        dt = self.dt
        if 'dt' in kwargs:
            dt = kwargs['dt']
        assert(dt is not None)

        self.ukf.predict(dt)
        super().propagate()

    def update(self, observation: np.ndarray) -> None:
        self.ukf.update(observation)
        super().update()

    @property
    def state(self) -> Any:
        return self.ukf.x

    @property
    def state_covar(self) -> Any:
        return self.ukf.P

    @property
    def obs_covar(self) -> Any:
        return self.ukf.R

    @property
    def process_noise(self) -> Any:
        return self.ukf.Q
