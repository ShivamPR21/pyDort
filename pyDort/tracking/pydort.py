import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from clort.clearn.data.dataframe import ArgoverseObjectDataFrame
from pyDort.representation.base import DataRepresentation
from pyDort.sem.data_association import DataAssociation
from pyDort.sem.evolution import StateEvolution

from .tracks import Track3D, TrackStatus
from .transform_utils import bbox_3d_from_8corners


class PyDort:

    def __init__(self, max_age : int, dt : float, min_hits : int,
                 appearance_model : Union[Type[DataRepresentation], DataRepresentation],
                 da_model : Union[Type[DataAssociation], DataAssociation],
                 sem : Callable[..., Union[Type[StateEvolution], StateEvolution]],
                 config_file : str,
                 track_type : Callable[..., Union[Type[Track3D], Track3D]] = Track3D) -> None:
        self.max_age = max_age
        self.dt = dt
        self.min_hits = min_hits
        self.track_type = track_type
        self.tracks : List[track_type] = []
        self.frame_count = 0
        self.appearance_model = appearance_model
        self.da_model = da_model
        self.sem = sem
        self.config_file = open(config_file)
        self.conf : Dict[str, Union[Dict, Any]] = json.load(self.config_file)

        self.ocmt : Dict[str, str] = self.conf["obj_class-motion_type"]
        self.ocmt_rev_map = {value: key for key, value in self.conf["obj_class-motion_type"].items()}

    def update(self, dets_all : List[ArgoverseObjectDataFrame]) -> List[List[Any]]:
        self.frame_count += 1
        state_dims = self.track_type().tracklet_class().state_dims
        n_tracks = len(self.tracks)
        print(f'Number of Detections : {len(dets_all)}, and Number of Tracks : {n_tracks}')

        trks_dsc : Union[List[np.ndarray], np.ndarray] = []         # N x d , #get descriptor for tracks to be attached.
        trks_st : Union[List[np.ndarray], np.ndarray] = []         # N x d , #get descriptor for tracks to be attached.
        ret : List[np.ndarray] = []         # N x 9, # N x [Cx, Cy, Cz, phi, l, b, h, id, class]


        for t in range(n_tracks):
            try:
                self.tracks[t].propagate()
            except:
                self.tracks[t].status = TrackStatus.Invalid
                continue

            dsc, st = self.tracks[t].descriptor(state_aug=True)
            dsc, st = dsc.reshape((-1, )), st.reshape((-1, ))

            trks_dsc += [dsc] # append descriptor to tracks descriptor list
            trks_st += [st] # append descriptor to tracks descriptor list

            if(np.any(np.isnan(dsc)) or (np.any(np.isnan(st)))):
                self.tracks[t].status = TrackStatus.Invalid

        # Remove descriptors, and trackers for invalid tracks
        trks_dsc = np.array(trks_dsc, dtype=np.float32) if len(trks_dsc) > 0 else np.empty((0, 1), dtype=np.float32)
        trks_st = np.array(trks_st, dtype=np.float32) if len(trks_st) > 0 else np.empty((0, 1), dtype=np.float32)

        trks_dsc, trks_st = np.ma.compress_rows(np.ma.masked_invalid(trks_dsc)), np.ma.compress_rows(np.ma.masked_invalid(trks_st))

        n_removed = self.remove_tracks_by_status(TrackStatus.Invalid)
        print(f'Number of invalid tracks removed: {n_removed}/{n_tracks}')

        dets_dsc, dets_st = self.feature_from_detections(dets_all, state_aug=True)
        dets_dsc = np.array(dets_dsc, dtype=np.float32) if len(dets_all) > 0 else np.empty((0, 1), dtype=np.float32)
        dets_st = np.array(dets_st, dtype=np.float32) if len(dets_all) > 0 else np.empty((0, 1), dtype=np.float32)

        print(dets_dsc.shape, trks_dsc.shape, dets_st.shape, trks_st.shape)
        matched, unmatched_dets, unmatched_trks = \
            self.da_model(dets_dsc = dets_dsc if self.da_model.descriptor_aug else None,
                          trks_dsc = trks_dsc if self.da_model.descriptor_aug else None,
                          dets_state = dets_st if self.da_model.state_aug else None,
                          trks_state = trks_st if self.da_model.state_aug else None,
                          scm_w = np.array([0.5, 0.5, 0.3], dtype=np.float32))

        print(f'Matchings -> {len(matched)}::{len(unmatched_dets)}::{len(unmatched_trks)}')

        # Update matched tracks
        for t,trk in enumerate(self.tracks):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:,1]==t)[0],0][0]     # a list of index
                bbox = bbox_3d_from_8corners(dets_all[d].bbox_global, dets_all[d].dims)
                try:
                    trk.update(bbox, dets_dsc[d, :])
                except:
                    trk.status = TrackStatus.Stale

        # Create new tracks for unmatched detections
        for i in unmatched_dets:        # a scalar of index
            obj_class = self.ocmt[dets_all[i].object_type]

            trk = self.track_type(obj_type=obj_class, dt=self.dt, stale_lim=self.max_age, sem=self.sem)

            state_dims, obs_dims = trk.tracklet_class().state_dims, trk.tracklet_class().obs_dims
            x_covar, z_covar, Q = None, None, None

            x_covar = np.array(self.conf[obj_class]['x_covar'], dtype=np.float32)
            x_covar = np.diag(x_covar) if len(x_covar) == state_dims else x_covar.reshape((state_dims, state_dims))

            z_covar = np.array(self.conf[obj_class]['z_covar'], dtype=np.float32)
            z_covar = np.diag(z_covar) if len(z_covar) == obs_dims else z_covar.reshape((obs_dims, obs_dims))

            Q = np.array(self.conf[obj_class]['Q'], dtype=np.float32)
            Q = np.diag(Q) if len(Q) == state_dims else Q.reshape((state_dims, state_dims))

            x_init = np.array(self.conf[obj_class]['x_init'], dtype=np.float32).reshape((state_dims, ))

            bbox = bbox_3d_from_8corners(dets_all[i].bbox_global, dets_all[i].dims)
            tracklet = trk.tracklet_class(state=None, observation=bbox, x_covar=x_covar, z_covar=z_covar, Q=Q, descriptor=dets_dsc[i, :])
            tracklet.update(state=tracklet.state_from_observation(tracklet.observation, x_init))

            trk.init(tracklet) # Initialize the track

            self.tracks.append(trk) # Add track to track list

        n_tracks = len(self.tracks)
        n_removed = self.remove_tracks_by_status(TrackStatus.Stale)
        print(f'Number of Stale tracks removed: {n_removed}/{n_tracks}, \nNumber of Tracks now : {len(self.tracks)}\n')

        # Need to get logging values
        for trk in self.tracks:
            if trk.tracklet_count >= self.min_hits:
                d = trk.track[-1].observation_from_state(trk.state)
                idx = trk.track_id
                obj_class = self.ocmt_rev_map[trk.obj_class]
                ret += [[*(d.tolist()), idx, obj_class]]

        return ret


    def remove_tracks_by_status(self, status:TrackStatus) -> int:
        n_tracks = len(self.tracks)
        n_removed = 0
        for i in reversed(range(n_tracks)):
            if self.tracks[i].status == status:
                self.tracks.pop(i)
                n_removed += 1
        return n_removed

    def feature_from_detections(self, detections : List[ArgoverseObjectDataFrame],
                                **kwargs) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self.appearance_model(detections, **kwargs)
