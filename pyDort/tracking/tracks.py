from enum import Enum
from typing import Any, Callable, List, Optional, Type, Union

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
                id : Optional[int] = None,
                obj_type : str = 'pedestrian-cv',
                dt : float = 0.5,
                stale_lim : int = 5,
                cache_lim : int = -1,
                sem : Optional[Callable[..., Type[StateEvolution]]] = None
                ) -> None:

        self.track_id = id if id is not None else Track3D.count
        Track3D.count += 1

        assert(obj_type in tracklets)
        self.obj_class = obj_type
        self.tracklet_class : Type[Tracklet] = tracklets[obj_type]
        self.dt = dt
        self.track : Union[Queue, List[Type[Tracklet]]] = Queue(maxsize=cache_lim)

        self.stale_lim = stale_lim
        self.status : Optional[TrackStatus] = None

        self.tracklet_count = 0
        self.missed_frames = 0
        self.updates_count = 0

        self.sem = sem

        self.dsc : List[Any] = [None, None]

    def init(self, tracklet: Type[Tracklet]):
        info = tracklet.info
        # info['alpha'] = np.zeros((tracklet.state_dims,))+1e-3
        info['alpha'] = 1e-1
        self.sem = self.sem(tracklet.state_dims, tracklet.obs_dims, self.dt, info)

        self.tracklet_count += 1
        self.track.put(tracklet)

        self.status = TrackStatus.Init

    def propagate(self, **kwargs) -> None:
        self.status = TrackStatus.Active

        if (self.sem.status == SemStatus.Propagated):
            self.updates_count = 0
            self.missed_frames += 1
            self.status = TrackStatus.Missing

        self.sem.propagate(**kwargs)

        tracklet = self.tracklet_class(
            self.sem.state,
            None,
            self.sem.state_covar,
            self.sem.obs_covar,
            self.sem.process_noise,
            None)

        tracklet.set_status(TrackletType.Dummy)
        if (self.stale_lim - self.missed_frames <= -1):
            print(f'Track id {self.track_id} stale : {self.stale_lim}/{self.missed_frames}')
            self.status = TrackStatus.Stale
            tracklet.set_status(TrackletType.End)

        self.tracklet_count += 1
        self.track.put(tracklet)

    def update(self, observation: np.ndarray):

        self.sem.update(observation)
        self.track[-1].update(self.sem.state,
                              observation,
                              self.sem.state_covar,
                              descriptor=None)

        self.missed_frames = 0
        self.updates_count += 1
        self.status = TrackStatus.Active
        self.track[-1].set_status(TrackletType.Real)

    @property
    def state(self) -> np.ndarray:
        return self.track[-1].state
