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

from typing import Any, Dict, Optional

import numpy as np
import ukfm

from .evolution import StateEvolution


class UKF(StateEvolution):

    def __init__(self,
                 state_dim: int = 3,
                 obs_dim: int = 3,
                 dt: float = None,
                 kwargs: Optional[Dict[str, Any]] = None) -> None:
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

    def propagate(self, **kwargs) -> None:
        dt = self.dt
        if 'dt' in kwargs: dt = kwargs['dt']
        assert(dt is not None)
        omega = kwargs['omega'] if 'omega' in kwargs else None

        self.ukf.propagation(omega, dt)
        super().propagate()

    def update(self, observation: np.ndarray) -> None:
        self.ukf.update(observation)
        super().update()

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
