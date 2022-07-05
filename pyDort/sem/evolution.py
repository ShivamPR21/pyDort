'''
Copyright (C) 2021  Shiavm Pandey

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
from typing import Any


class SemStatus(Enum):
    Init = 1
    Propagated = 2
    Updated = 3


class StateEvolution:

    def __init__(self,
                 state_dim : int = 3,
                 obs_dim : int = 3,
                 dt: float = None) -> None:
        self.state_dim, self.obs_dim = state_dim, obs_dim
        self.dt = dt
        self.status = SemStatus.Init

    def propagate(self) -> None:
        self.status = SemStatus.Propagated

    def update(self) -> None:
        self.status = SemStatus.Updated

    @property
    def state(self) -> Any:
        raise NotImplementedError

    @property
    def state_covar(self) -> Any:
        raise NotImplementedError

    @property
    def obs_covar(self) -> Any:
        raise NotImplementedError

    @property
    def process_noise(self) -> Any:
        raise NotImplementedError
