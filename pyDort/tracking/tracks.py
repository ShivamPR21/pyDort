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
from typing import Callable, List, Optional, Tuple, Type, Union

import numpy as np

from ..helpers import Queue
from ..sem.evolution import SemStatus, StateEvolution
from .info import tracklets
from .tracklet import Tracklet, TrackletType


class TrackStatus(Enum):
    Init = 1
    Active = 2
    Missing = 3
    Stale = 4
    Invalid = 5


class Track3D:

    count = 0

    def __init__(self,
                id : int = None,
                obj_type : str = 'pedestrian-cv',
                dt : float = 0.5,
                stale_lim : int = 5,
                cache_lim : int = -1,
                sem : Callable[..., Type[StateEvolution]] = None) -> None:

        if id is None:
            self.track_id = Track3D.count
            Track3D.count += 1
        else:
            self.track_id = id

        assert(obj_type in tracklets)
        self.obj_class = obj_type
        self.tracklet_class : Type[Tracklet] = tracklets[obj_type]
        self.dt = dt
        self.track : Union[Queue, List[Type[Tracklet]]] = Queue(maxsize=cache_lim)

        self.stale_lim = stale_lim
        self.status : Optional[TrackStatus] = None

        self.tracklet_count = 0
        self.missed_frames = 0

        self.sem : Type[StateEvolution] = sem

    def init(self, tracklet: Type[Tracklet]):
        info = tracklet.info
        info['alpha'] = np.zeros((tracklet.state_dims,))+1e-3
        self.sem = self.sem(tracklet.state_dims, tracklet.obs_dims, self.dt, info)

        self.tracklet_count += 1
        self.track.put(tracklet)

        self.status = TrackStatus.Init

    def dsc_propagation(self) -> None:
        return self.descriptor(state_aug=False)

    def propagate(self, **kwargs) -> None:
        self.status = TrackStatus.Active

        if (self.sem.status == SemStatus.Propagated):
            self.missed_frames += 1
            self.status = TrackStatus.Missing

        self.sem.propagate(**kwargs)
        dsc, _ = self.dsc_propagation() # state being stored already in sem module

        tracklet = self.tracklet_class(
            self.sem.state,
            None,
            self.sem.state_covar,
            self.sem.obs_covar,
            self.sem.process_noise,
            dsc)

        self.tracklet_count += 1
        self.track.put(tracklet)

        if (self.stale_lim - self.missed_frames <= 0):
            self.status = TrackStatus.Stale
            tracklet.set_status(TrackletType.End)
        else:
            tracklet.set_status(TrackletType.Dummy)


    def update(self, observation: np.ndarray, descriptor: Optional[np.ndarray] = None):

        self.sem.update(observation)
        self.track[-1].update(self.sem.state,
                              observation,
                              self.sem.state_covar,
                              descriptor=descriptor)

        self.missed_frames = 0
        self.status = TrackStatus.Active
        self.track[-1].set_status(TrackletType.Real)

    def descriptor(self, state_aug : bool=True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:

        state, dsc = None, None

        if self.track[-1].descriptor is not None:
            dsc = self.track[-1].descriptor.reshape((-1, 1))

        if state_aug:
            state = self.state[:7].reshape((-1, 1))
        return dsc, state

    @property
    def state(self) -> np.ndarray:
        return self.track[-1].state
