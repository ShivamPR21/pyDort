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
from typing import Callable, Type, Union, float, int

import numpy as np
from sem.evolution import StateEvolution

from ..helpers import Queue
from .info import tracklets
from .primitives import Primitives
from .tracklet import Tracklet


class TrackStatus(Enum):
    Init = 1
    Active = 2
    Stale = 3
    Missing = 4


class Track3D:

    def __init__(self,
                id : int = None,
                obj_type : str = 'pedestrian-cv',
                dt : float = 0.5,
                stale_lim : int = 5,
                cache_lim : int = 1000,
                sem : Callable[..., Type[StateEvolution]] = None) -> None:

        assert(id)
        self.track_id = id
        self.tracklet_class : Type[Primitives] = tracklets[obj_type]
        self.dt = dt
        self.track = Queue(maxsize=cache_lim)

        self.stale_lim = stale_lim
        self.status = TrackStatus.Init

        self.tracks_count = 0
        self.missed_frames = 0

        self.sem : Type[StateEvolution] = sem

    def init(self, tracklet: Type[Tracklet]):
        info = tracklet.info
        info['alpha'] = 0.1
        self.sem = self.sem(len(tracklet.state_dims), len(tracklet.obs_dims), self.dt, info)

    def propagate(self, **kwargs) -> None:
        self.sem.predict(**kwargs)
        tracklet = self.tracklet_class(
            self.sem.state,
            None,
            self.sem.state_covar,
            self.sem.obs_covar,
            self.sem.process_noise)
        self.track.put(tracklet)

    def update(self, observation: np.ndarray):
        self.sem.update(observation)
        self.track[-1].update(self.sem.state,
                              observation,
                              self.sem.state_covar)
