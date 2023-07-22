import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from pyDort.sem.data_association import get_distance, get_distance_iou_3d
from pyDort.sem.evolution import StateEvolution

from .tracks import Track3D, TrackStatus
from .transform_utils import batch_bbox_3d_from_8corners, bbox_3d_from_8corners


class PyDort:

    def __init__(self, max_age : int, dt : float, min_hits : int,
                 sem : Callable[..., Type[StateEvolution] | StateEvolution],
                 config_file : str,
                 track_type : Callable[..., Type[Track3D] | Track3D] = Track3D,
                 rep_update: str = 'replace', # momentum, similarity
                 Q: int = 1,
                 alpha_thresh: Tuple[float, float] = (0.1, 0.9),
                 beta_thresh: Tuple[float, float] = (0.1, 0.9),
                 state_w: List[float] = [0.3, 0.3, 0.3],
                 dsc_w: List[float] = [0.3, 0.3],
                 cm_fusion_w: List[float] = [0.5, 0.5],
                 trks_center_w: List[float] = [0.5],
                 matching_threshold: float = 0.8) -> None:
        assert(len(trks_center_w) == Q)

        self.max_age = max_age
        self.dt = dt
        self.min_hits = min_hits
        self.track_type = track_type
        self.tracks : List[Any] = []
        self.frame_count = 0
        self.sem = sem
        self.rep_update = rep_update
        self.q = Q
        self.state_w = state_w
        self.dsc_w = dsc_w
        self.trks_center_w = trks_center_w
        self.cm_fusion_w = cm_fusion_w
        self.a_t = alpha_thresh
        self.b_t = beta_thresh
        self.matching_threshold = matching_threshold
        self.config_file = open(config_file)
        self.conf : Dict[str, Union[Dict, Any]] = json.load(self.config_file)

        self.ocmt : Dict[str, str] = self.conf["obj_class-motion_type"]
        self.ocmt_rev_map = {value: key for key, value in self.conf["obj_class-motion_type"].items()}

    def update(self, bboxs: np.ndarray, reprs: List[np.ndarray | None], _obj_cls: List[str]) -> List[List[Any]]:
        # self.tracks = [] # Uncomment to see detections #TODO: Be cautious
        self.frame_count += 1
        state_dims = self.track_type().tracklet_class().state_dims # type: ignore
        n_tracks = len(self.tracks)
        print(f'Number of Detections : {len(bboxs)}, and Number of Tracks : {n_tracks}')

        trks_dsc1 : List[np.ndarray] | np.ndarray | None = []         # N x d , #get descriptor for tracks to be attached.
        trks_dsc2 : List[np.ndarray] | np.ndarray | None = []         # N x d , #get descriptor for tracks to be attached.
        trks_st : List[np.ndarray] | np.ndarray = []         # N x d , #get descriptor for tracks to be attached.
        ret : List[Any] = []         # N x 9, # N x [Cx, Cy, Cz, phi, l, b, h, id, class]

        for t in range(n_tracks):
            try:
                self.tracks[t].propagate()
            except RuntimeError:
                self.tracks[t].status = TrackStatus.Invalid
                continue

            dsc1, dsc2 = self.tracks[t].dsc
            st = self.tracks[t].state[:7]

            dsc1 = np.expand_dims(dsc1, axis=0) if dsc1 is not None else None
            dsc2 = np.expand_dims(dsc2, axis=0) if dsc2 is not None else None
            st = np.expand_dims(st, axis=0)

            if (dsc1 is not None and np.any(np.isnan(dsc1))):
                self.tracks[t].status = TrackStatus.Invalid
            elif (dsc2 is not None and np.any(np.isnan(dsc2))):
                self.tracks[t].status = TrackStatus.Invalid
            elif np.any(np.isnan(st)):
                self.tracks[t].status = TrackStatus.Invalid
            else:
                pass

            if self.tracks[t].status == TrackStatus.Invalid:
                continue

            if dsc1 is not None:
                trks_dsc1.append(dsc1) # append descriptor to tracks descriptor list
            if dsc2 is not None:
                trks_dsc2.append(dsc2) # append descriptor to tracks descriptor list
            trks_st += [st] # append descriptor to tracks descriptor list

        n_removed = self.remove_tracks_by_status(TrackStatus.Invalid)
        print(f'Number of invalid tracks removed: {n_removed}/{n_tracks}')

        trks_dsc1 = np.concatenate(trks_dsc1, axis=0) if trks_dsc1 != [] else None
        trks_dsc2 = np.concatenate(trks_dsc2, axis=0) if trks_dsc2 != [] else None
        trks_st = np.concatenate(trks_st, axis=0) if trks_st != [] else np.empty((0, 1))

        assert(len(reprs) == 2)
        dets_dsc1, dets_dsc2 = reprs
        bboxs_3d = batch_bbox_3d_from_8corners(bboxs)

        M, N = bboxs_3d.shape[0], trks_st.shape[0]

        # Compute cost matrix
        ## State cost matrix
        cm_centroid = self.cost_matrix_state(M, N, bboxs_3d, trks_st, 'centroid')
        cm_iou = self.cost_matrix_state(M, N, bboxs_3d, trks_st, 'iou')
        cm_yaw = self.cost_matrix_state(M, N, bboxs_3d, trks_st, 'orientation')
        cm_state = (self.state_w[0] * cm_centroid + self.state_w[1] * cm_iou + self.state_w[2] * cm_yaw)/np.sum(self.state_w)

        ## Representation cost matrix
        dsc_cm = None
        dsc_cm1 = self.cost_matrix_des(M, N, dets_dsc1, trks_dsc1) if dets_dsc1 is not None and trks_dsc1 is not None else None
        dsc_cm2 = self.cost_matrix_des(M, N, dets_dsc2, trks_dsc2) if dets_dsc2 is not None and trks_dsc2 is not None else None

        if dsc_cm1 is not None and dsc_cm2 is not None:
            dsc_cm = (self.dsc_w[0] * dsc_cm1 + self.dsc_w[1] * dsc_cm2)/np.sum(self.dsc_w)
        elif dsc_cm1 is not None:
            dsc_cm = dsc_cm1
        elif dsc_cm2 is not None:
            dsc_cm = dsc_cm2
        else:
            pass

        ## Final Cost matrix
        cost_matrix = None
        if dsc_cm is not None:
            cost_matrix = self.cm_fusion_w[0] * cm_state + self.cm_fusion_w[1] * dsc_cm
            cost_matrix /= np.sum(self.cm_fusion_w)
        else:
            cost_matrix = cm_state

        matched, unmatched_dets, unmatched_trks =  self.solve(M, N, cost_matrix) # type: ignore

        print(f'Matchings -> {len(matched)}::{len(unmatched_dets)}::{len(unmatched_trks)}')

        # Update matched tracks
        for t, trk in enumerate(self.tracks):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:,1]==t)[0],0][0]     # a list of index
                try:
                    trk.update(bboxs_3d[d, :])
                    trk.dsc[0] = self.get_updated_dsc(dets_dsc1[d, :, :], trks_dsc1[t, :, :]) if dets_dsc1 is not None and trks_dsc1 is not None else None
                    trk.dsc[1] = self.get_updated_dsc(dets_dsc2[d, :, :], trks_dsc2[t, :, :]) if dets_dsc2 is not None and trks_dsc2 is not None else None
                except RuntimeError:
                    # Put descriptor to unmatched detections
                    unmatched_dets = np.append(unmatched_dets, [d])
                    print(f'Unable to propagate the track {trk.track_id} :: Num tracklets : {trk.tracklet_count}, Num missed frames : {trk.missed_frames}, Num updates : {trk.updates_count}')
                    trk.status = TrackStatus.Stale

        n_tracks = len(self.tracks)
        n_removed = self.remove_tracks_by_status(TrackStatus.Stale)
        print(f'Not able to update tracks removed: {n_removed}/{n_tracks}, \nNumber of Tracks now : {len(self.tracks)}')

        # Create new tracks for unmatched detections
        for i in unmatched_dets:        # a scalar of index
            obj_class = self.ocmt[_obj_cls[i]]

            trk : Track3D = self.track_type(obj_category=_obj_cls[i], obj_type=obj_class, dt=self.dt, stale_lim=self.max_age, sem=self.sem) # type: ignore

            state_dims, obs_dims = trk.tracklet_class().state_dims, trk.tracklet_class().obs_dims # type: ignore
            x_covar, z_covar, Q = None, None, None

            x_covar = np.array(self.conf[obj_class]['x_covar'], dtype=np.float32)
            x_covar = np.diag(x_covar) if len(x_covar) == state_dims else x_covar.reshape((state_dims, state_dims))

            z_covar = np.array(self.conf[obj_class]['z_covar'], dtype=np.float32)
            z_covar = np.diag(z_covar) if len(z_covar) == obs_dims else z_covar.reshape((obs_dims, obs_dims))

            Q = np.array(self.conf[obj_class]['Q'], dtype=np.float32)
            Q = np.diag(Q) if len(Q) == state_dims else Q.reshape((state_dims, state_dims))

            x_init = np.array(self.conf[obj_class]['x_init'], dtype=np.float32).reshape((state_dims, ))

            tracklet = trk.tracklet_class(state=None, observation=bboxs_3d[i, :],
                                          x_covar=x_covar, z_covar=z_covar, Q=Q)
            tracklet.update(state=tracklet.state_from_observation(tracklet.observation, x_init))

            trk.init(tracklet) # type: ignore # Initialize the track
            trk.dsc[0] = dets_dsc1[i, :, :] if dets_dsc1 is not None else None
            trk.dsc[1] = dets_dsc2[i, :, :] if dets_dsc2 is not None else None

            self.tracks.append(trk) # Add track to track list

        n_tracks = len(self.tracks)
        n_removed = self.remove_tracks_by_status(TrackStatus.Stale)
        print(f'Number of Stale tracks removed: {n_removed}/{n_tracks}, \nNumber of Tracks now : {len(self.tracks)}\n')

        # Need to get logging values
        for trk in self.tracks:
            if trk.tracklet_count >= self.min_hits:
                d = trk.track[-1].observation_from_state(trk.state).tolist() # type: ignore
                d.append(trk.track_id)
                d.append(trk.obj_category)
                ret.append(d)

        return ret


    def get_updated_dsc(self, dets_dsc: np.ndarray, trks_dsc: np.ndarray):
        # trks_dsc -> [q, n]
        # dets_dsc -> [n]
        if dets_dsc.ndim == 1:
            dets_dsc = dets_dsc.reshape((-1, 1))

        if self.rep_update == 'replace':
            return dets_dsc
        elif self.rep_update == 'momentum':
            return (0.7*dets_dsc + 0.3*trks_dsc)
        elif self.rep_update == 'similarity':
            reps = trks_dsc
            alpha = np.clip(dets_dsc @ trks_dsc, self.a_t[0], self.a_t[1]) # [q, 1]

            for q in range(self.q):
                if q == 0:
                    reps[q, :] = (1. - alpha[q]) * trks_dsc[q, :] + alpha[q] * dets_dsc[:, 0]
                else:
                    sf = max(q+1, 3)
                    beta = np.clip(trks_dsc[q-1, :] @ trks_dsc[q, :], self.b_t[0], self.b_t[1])
                    reps[q, :] = (1. - (alpha[q] + beta)/sf) * trks_dsc[q, :] + (beta/sf) * trks_dsc[q-1, :] + (alpha[q]/sf) * dets_dsc[:, 0]

            reps /= (np.linalg.norm(reps, axis=1, keepdims=True) + 1e-9)
            return reps
        else:
            raise NotImplementedError(f'Representation update method {self.rep_update} isn\'t  implemented.')

    def cost_matrix_des(self, M:int, N:int, dets:np.ndarray, trks:np.ndarray) -> np.ndarray:
        # dets -> [M, n]
        # trks -> [N, q, n]
        cost_m = np.zeros((M, N), dtype=np.float32)

        for q in range(self.q):
            cost_m -= (dets @ trks[:, q, :].T) * self.trks_center_w[q]

        cost_m /= np.sum(self.trks_center_w)

        # Normalize
        if not (M == 0 or N == 0):
            cost_m -= cost_m.min()
            cost_m /= (abs(cost_m.max())+1e-9)

        return cost_m

    def cost_matrix_state(self, M:int, N:int, dets:np.ndarray, trks:np.ndarray, distance:str) -> np.ndarray:
        cost_m = np.zeros((M, N), dtype=np.float32)

        for m in range(M):
            for n in range(N):
                cost_m[m, n] = get_distance(dets[m, :], trks[n, :], distance)

        # Normalize
        if not (M == 0 or N == 0):
            cost_m -= cost_m.min()
            cost_m /= (abs(cost_m.max())+1e-9)

        return cost_m

    def remove_tracks_by_status(self, status:TrackStatus) -> int:
        n_tracks = len(self.tracks)
        n_removed = 0
        for i in reversed(range(n_tracks)):
            if self.tracks[i].status == status:
                self.tracks.pop(i)
                n_removed += 1
        return n_removed

    def solve(self, M:int, N:int, cm:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert(cm.ndim == 2)

        # Handle zero detections or tracks
        if M == 0 or N == 0:
            return np.empty((0,2), dtype=np.int32), np.arange(M, dtype=np.int32), np.arange(N, dtype=np.int32)

        # Normalize
        cm -= cm.min()
        cm /= (abs(cm.max())+1e-9)

        row_idxs, col_idxs = linear_sum_assignment(cm)

        unmatched_detections = []
        for d in range(M):
            if(d not in row_idxs):
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t in range(N):
            if(t not in col_idxs):
                unmatched_trackers.append(t)

        # similarity matrix : negative of cost_matrix
        sim_matrix = 1 - cm # Similarity range (0, 1]

        #filter out matched with low IOU
        matches = []
        for m, n in zip(row_idxs, col_idxs, strict=True):
            if(sim_matrix[m, n]<self.matching_threshold):
                unmatched_detections.append(m)
                unmatched_trackers.append(n)
            else:
                matches.append(np.array([[m, n]], dtype=np.int32))

        if(len(matches)==0):
            matches = np.empty((0,2), dtype=np.int32)
        else:
            matches = np.concatenate(matches, axis=0)

        return np.array(matches, dtype=np.int32), np.array(unmatched_detections, dtype=np.int32), np.array(unmatched_trackers, dtype=np.int32)
