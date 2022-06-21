'''
Copyright (C) 2022  Shiavm Pandey

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

from enum import Enum
from typing import Any, Callable, Optional, Type, Union

import numpy as np
import ukfm

from .evolution import StateEvolution


class SemStatus(Enum):
    Init = 1
    Propagated = 2
    Updated = 3


class UKF(StateEvolution):

    def __init__(self,
                 state_dim: int = 3,
                 obs_dim: int = 3,
                 dt: float = None,
                 **kwargs) -> None:
        super().__init__(state_dim, obs_dim, dt)

        self.ukf = ukfm.UKF(kwargs['f'],
                            kwargs['h'],
                            kwargs['phi'],
                            kwargs['phi_inv'],
                            kwargs['Q'],
                            kwargs['R'],
                            kwargs['alpha'],
                            kwargs['state0'],
                            kwargs['P0'])

        self.status = SemStatus.Init

    def propagate(self, **kwargs) -> None:
        dt = self.dt
        if 'dt' in kwargs: dt = kwargs['dt']
        assert(dt is not None)
        if 'omega' in kwargs: omega = kwargs['omega']

        self.ukf.propagation(omega, dt)
        self.status = SemStatus.Propagated

    def update(self, observation: np.ndarray) -> None:
        self.ukf.update(observation)
        self.status = SemStatus.Updated

    @property
    def state(self) -> Any:
        return self.ukf.state

    @property
    def state_covar(self) -> Any:
        return self.ukf.P

    @property
    def obs_covar(self) -> Any:
        return self.ukf.R

    @property
    def process_noise(self) -> Any:
        return self.ukf.Q
