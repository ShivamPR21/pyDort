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
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..helpers import wrap_angle


def get_distance_iou_3d(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Note this is not traditional 2d or 3d iou, but rather we align two cuboids
    along their x-axes, and compare 3d volume differences.
    """
    w1 = x1[1]
    l1 = x1[0]
    h1 = x1[2]

    w2 = x2[1]
    l2 = x2[0]
    h2 = x2[2]

    x_overlap = max(0, min(l1 / 2, l2 / 2) - max(-l1 / 2, -l2 / 2))
    y_overlap = max(0, min(w1 / 2, w2 / 2) - max(-w1 / 2, -w2 / 2))
    overlapArea = x_overlap * y_overlap
    inter = overlapArea * min(h1, h2)
    union = w1 * l1 * h1 + w2 * l2 * h2 - inter
    score = 1 - inter / union

    return float(score)


def get_distance(x1: np.ndarray, x2: np.ndarray, name: str) -> float:
    """Get the distance between two poses, returns nan if distance is larger than detection threshold.

    Args:
        x1: first pose
        x2: second pose
        name: name of the field to test

    Returns:
        A distance value or NaN
    """
    if name == "centroid":
        dist = float(np.linalg.norm(x1[0:3] - x2[0:3]))
        # return dist if dist < 50 else float(np.nan)
        return dist
    elif name == "iou":
        return get_distance_iou_3d(x1[4:7], x2[4:7])
    elif name == "orientation":
        theta = np.array([x1[3]] - x2[3])
        dist = wrap_angle(theta).item()

        # Convert to degrees.
        return float(np.rad2deg(dist))
    else:
        raise NotImplementedError("Not implemented..")

class DataAssociation:

    def __init__(self, matching_threshold:float = 1e-1,
                 descriptor_aug:bool = True,
                 state_aug:bool = True,
                 dsc_cmf:Optional[Callable[..., np.ndarray]] = None,
                 state_cmf:Optional[Callable[..., np.ndarray]] = None,
                 dsc_st_w:Optional[np.ndarray] = None) -> None:
        if not descriptor_aug:
            state_aug = True

        if dsc_st_w is None:
            dsc_st_w = np.ones((2, ), dtype=np.float32)/2
        else:
            dsc_st_w = dsc_st_w.reshape((-1, ))
            assert(dsc_st_w.shape[0] == 2)
            dsc_st_w /= dsc_st_w.sum()+1e-9

        self.dsc_st_w = dsc_st_w
        # if descriptor_aug: assert(dsc_cmf is not None)
        # if state_aug: assert(state_cmf is not None)

        self.matching_threshold = matching_threshold
        self.descriptor_aug = descriptor_aug
        self.state_aug = state_aug

        self.dsc_cmf, self.state_cmf = dsc_cmf, state_cmf

    def __call__(self, dets_dsc:Optional[np.ndarray] = None, trks_dsc:Optional[np.ndarray] = None,
            dets_state:Optional[np.ndarray] = None, trks_state:Optional[np.ndarray] = None,
            **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        M, N, M_dsc, N_dsc, M_st, N_st = -1, -1, -1, -1, -1, -1

        if self.descriptor_aug:
            assert(dets_dsc is not None and trks_dsc is not None)
            (M_dsc, N_dsc) = (dets_dsc.shape[0], trks_dsc.shape[0])
            M, N = M_dsc, N_dsc

        if self.state_aug:
            assert(dets_state is not None and trks_state is not None)
            (M_st, N_st) = (dets_state.shape[0], trks_state.shape[0])
            M, N = M_st, N_st

        if self.state_aug and self.descriptor_aug:
            print((M_dsc, N_dsc) , (M_st, N_st))
            assert ((M_dsc, N_dsc) == (M_st, N_st))

        # Handle zero detections or tracks
        if M == 0 or N == 0:
            return np.empty((0,2), dtype=np.int32), np.arange(M, dtype=np.int32), np.arange(N, dtype=np.int32)

        overall_cm, state_cm, dsc_cm = np.zeros((M, N), dtype=np.float32), None, None

        if self.state_aug:
            state_cm = self.state_aug_cost_matrix(dets_state, trks_state, **kwargs)
            overall_cm += state_cm*self.dsc_st_w[1]

        if self.descriptor_aug:
            dsc_cm = self.descriptor_cost_matrix(dets_dsc, trks_dsc, **kwargs)
            overall_cm += dsc_cm*self.dsc_st_w[0]

        return self.solve(M, N, overall_cm)

    def state_aug_cost_matrix(self, dets_state:np.ndarray, trks_state:np.ndarray, **kwargs) -> np.ndarray:
        if self.state_cmf is not None:
            return self.state_cmf(dets_state, trks_state, **kwargs)

        return self.state_data_association_costmatrix(dets_state, trks_state, **kwargs)

    def descriptor_cost_matrix(self, dets_dsc:np.ndarray, trks_dsc:np.ndarray, **kwargs) -> np.ndarray:
        if self.dsc_cmf is not None:
            return self.dsc_cmf(dets_dsc, trks_dsc, **kwargs)

        return self.dsc_data_association_costmatrix(dets_dsc, trks_dsc, **kwargs)

    @staticmethod
    def state_data_association_costmatrix(dets : np.ndarray,
                                      trks : np.ndarray,
                                      scm_w : Optional[np.ndarray] = None) -> Union[Any, np.ndarray]:
        if scm_w is None:
            scm_w = np.ones((3, ), dtype=np.float32)/3
        else:
            scm_w = scm_w.reshape((-1, ))
            assert(scm_w.shape[0] == 3)
            scm_w /= np.sum(scm_w)+1e-9

        # dets : [M, n]
        # trks : [N, n]
        if len(trks) == 0 or len(dets) == 0:
            return np.empty((0,2), dtype=np.int32), np.arange(len(dets), dtype=np.int32), np.arange(len(trks), dtype=np.int32)

        assert(dets.shape[1] == trks.shape[1])

        M, N = dets.shape[0], trks.shape[0]

        cost_matrix_c : np.ndarray = np.zeros((M, N))
        cost_matrix_i : np.ndarray = np.zeros((M, N))
        cost_matrix_o : np.ndarray = np.zeros((M, N))
        for d in range(M):
            for t in range(N):
                cost_matrix_c[d, t] = get_distance(dets[d, :], trks[t, :], 'centroid')
                cost_matrix_i[d, t] = get_distance(dets[d, :], trks[t, :], 'iou')
                cost_matrix_o[d, t] = get_distance(dets[d, :], trks[t, :], 'orientation')

        cost_matrix_c /= cost_matrix_c.max()+1e-9
        cost_matrix_i /= cost_matrix_i.max()+1e-9
        cost_matrix_o /= cost_matrix_o.max()+1e-9

        cost_matrix = cost_matrix_c*scm_w[0] + cost_matrix_i*scm_w[1] + cost_matrix_o*scm_w[2]

        return cost_matrix

    @staticmethod
    def dsc_data_association_costmatrix(dets : np.ndarray,
                                      trks : np.ndarray,
                                      **kwargs) -> np.ndarray:
        # dets : [M, n]
        # trks : [N, n]
        dets /= (np.linalg.norm(dets, axis=1, keepdims=True)+1e-11)
        trks /= (np.linalg.norm(trks, axis=1, keepdims=True)+1e-11)
        # dets, trks = np.expand_dims(dets, axis=1), np.expand_dims(trks, axis=0)
        # cost_matrix = np.linalg.norm(dets - trks, axis=2)

        cost_matrix : np.ndarray = 1-dets.dot(trks.T) # [M, N] matrix
        cost_matrix /= (abs(cost_matrix.max())+1e-11)

        return cost_matrix

    def solve(self, M:int, N:int, cm:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert(cm.ndim == 2)
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
        for m, n in zip(row_idxs, col_idxs):
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
