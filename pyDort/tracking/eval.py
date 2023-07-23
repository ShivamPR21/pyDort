# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
import glob
import logging
import os
import pathlib
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union

import motmetrics as mm
import numpy as np
import pandas as pd
import wandb
from omegaconf import DictConfig
from shapely.geometry.polygon import Polygon

from pyDort.eval_utils import label_to_bbox
from pyDort.helpers import read_json_file, wrap_angle

mh = mm.metrics.create()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_PathLike = Union[str, "os.PathLike[str]"]

"""
Computes multiple object tracking (MOT) metrics. Please refer to the following
papers for definitions of the following metrics:

#FRAG, #IDSW: Milan et al., MOT16, https://arxiv.org/pdf/1603.00831.pdf
MT, ML: Leal-Taixe et al., MOT15, https://arxiv.org/pdf/1504.01942.pdf
MOTA: Bernardin et al. https://link.springer.com/article/10.1155/2008/246309
"""


def in_distance_range_pose(ego_center: np.ndarray, pose: np.ndarray, d_min: float, d_max: float) -> bool:
    # """Determine whether a pose in the ego-vehicle frame falls within a specified distance range
    #     of the egovehicle's origin.

    # Args:
    #     ego_center: ego center pose (zero if bbox is in ego frame).
    #     pose:  pose to test.
    #     cfg.d_min: minimum distance range
    #     cfg.d_max: maximum distance range

    # Returns:
    #     A boolean saying if input pose is with specified distance range.
    # """
    dist = float(np.linalg.norm(pose[0:3] - ego_center[0:3]))

    return dist > d_min and dist < d_max


def iou_polygon(poly1: Polygon, poly2: Polygon) -> float:
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return float(1 - inter / union)


def get_distance_iou_3d(x1: Dict[str, np.ndarray], x2: Dict[str, np.ndarray], name: str = "bbox") -> float:
    """
    Note this is not traditional 2d or 3d iou, but rather we align two cuboids
    along their x-axes, and compare 3d volume differences.
    """
    w1 = x1["width"]
    l1 = x1["length"]
    h1 = x1["height"]

    w2 = x2["width"]
    l2 = x2["length"]
    h2 = x2["height"]

    x_overlap = max(0, min(l1 / 2, l2 / 2) - max(-l1 / 2, -l2 / 2))
    y_overlap = max(0, min(w1 / 2, w2 / 2) - max(-w1 / 2, -w2 / 2))
    overlapArea = x_overlap * y_overlap
    inter = overlapArea * min(h1, h2)
    union = w1 * l1 * h1 + w2 * l2 * h2 - inter
    score = 1 - inter / union

    return float(score)


def get_distance(x1: Dict[str, np.ndarray], x2: Dict[str, np.ndarray], name: str) -> float:
    # """Get the distance between two poses, returns nan if distance is larger than detection threshold.

    # Args:
    #     x1: first pose
    #     x2: second pose
    #     name: name of the field to test

    # Returns:
    #     A distance value or NaN
    # """
    if name == "centroid":
        dist = float(np.linalg.norm(x1[name][0:3] - x2[name][0:3]))
        return dist if dist < 2 else float(np.nan)
    elif name == "iou":
        return get_distance_iou_3d(x1, x2, name)
    elif name == "orientation":
        theta = np.array([x1["orientation"]] - x2["orientation"])
        dist = wrap_angle(theta).item()

        # Convert to degrees.
        return float(np.rad2deg(dist))
    else:
        raise NotImplementedError("Not implemented..")

class IntIdFromString:

    def __init__(self) -> None:
        self.n_elements = 0
        self.dct : Dict[str, int] = {}

    def get_id(self, uid: str) -> int:
        if uid in self.dct.keys():
            return self.dct[uid]

        self.dct.update({uid: self.n_elements})
        self.n_elements += 1
        return self.n_elements-1

def eval_tracks(cfg: DictConfig) -> Dict[str, Any]:
    # """Evaluate tracking output.

    # Args:
    #     cfg.prediction_path: path to tracker output root, containing log_id subdirs
    #     cfg.gt_path: path to dataset root, containing log_id subdirs
    #     cfg.d_min: minimum allowed distance range for ground truth and predicted objects,
    #         in meters
    #     cfg.d_max: maximum allowed distance range, as above, in meters
    #     out_file: output file object
    #     centroid_method: method for ground truth centroid estimation
    #     cfg.diffatt: difficulty attribute ['easy',  'far', 'fast', 'occ', 'short']. Note that if
    #         tracking according to a specific difficulty attribute is computed, then all ground
    #         truth annotations not fulfilling that attribute specification are
    #         disregarded/dropped out. Since the corresponding track predictions are not dropped
    #         out, the number of false positives, false negatives, and MOTA will not be accurate
    #         However, `mostly tracked` and `mostly lost` will be accurate.
    #     cfg.categories: such as "VEHICLE" "PEDESTRIAN"
    # """
    acc_c = mm.MOTAccumulator(auto_id=True)
    acc_i = mm.MOTAccumulator(auto_id=True)
    acc_o = mm.MOTAccumulator(auto_id=True)

    ID_gt_all: List[str] = []

    count_all: int = 0
    uid_to_int = IntIdFromString()

    logger.info(f'{cfg.categories = }')

    if cfg.diffatt is not None:

        pkl_path = os.path.join(os.path.dirname(__file__), "argo_assets", "dict_att_all.pkl")
        if not os.path.exists(pkl_path):
            # generate them on the fly
            logger.info(pkl_path)
            raise NotImplementedError

        pickle_in = open(pkl_path, "rb")  # open(f"{cfg.gt_path}/dict_att_all.pkl","rb")
        dict_att_all = pickle.load(pickle_in)

    path_datasets = glob.glob(os.path.join(cfg.gt_path, "*"))
    num_total_gt = 0

    for path_dataset in path_datasets:

        log_id = pathlib.Path(path_dataset).name
        if len(log_id) == 0 or log_id.startswith("_"):
            continue

        path_tracker_output = os.path.join(cfg.prediction_path, log_id)

        path_track_data = sorted(
            glob.glob(os.path.join(os.fspath(path_tracker_output), "per_sweep_annotations_amodal", "*"))
        )

        logger.info("log_id = %s", log_id)

        for ind_frame in range(len(path_track_data)):
            if ind_frame % 50 == 0:
                logger.info("%d/%d" % (ind_frame, len(path_track_data)))

            timestamp_lidar = int(Path(path_track_data[ind_frame]).stem.split("_")[-1])
            path_gt = os.path.join(
                path_dataset,
                "per_sweep_annotations_amodal",
                f"tracked_object_labels_{timestamp_lidar}.json",
            )

            if not os.path.exists(path_gt):
                logger.warning("Missing ", path_gt)
                continue

            gt_data = read_json_file(path_gt)

            gt: Dict[str, Dict[str, Any]] = {}
            id_gts = []
            for i in range(len(gt_data)):

                if gt_data[i]["label_class"] not in cfg.categories:
                    continue

                if cfg.diffatt is not None:

                    if cfg.diffatt not in dict_att_all["test"][log_id][gt_data[i]["track_label_uuid"]]["difficult_att"]:
                        continue

                bbox, orientation = label_to_bbox(gt_data[i])

                center = np.array(
                    [
                        gt_data[i]["center"]["x"],
                        gt_data[i]["center"]["y"],
                        gt_data[i]["center"]["z"],
                    ]
                )
                if bbox[3] > 0 and in_distance_range_pose(np.zeros(3), center, cfg.d_min, cfg.d_max):
                    track_label_uuid = gt_data[i]["track_label_uuid"]
                    gt[track_label_uuid] = {}
                    gt[track_label_uuid]["centroid"] = center

                    gt[track_label_uuid]["bbox"] = bbox
                    gt[track_label_uuid]["orientation"] = orientation
                    gt[track_label_uuid]["width"] = gt_data[i]["width"]
                    gt[track_label_uuid]["length"] = gt_data[i]["length"]
                    gt[track_label_uuid]["height"] = gt_data[i]["height"]

                    if track_label_uuid not in ID_gt_all:
                        ID_gt_all.append(track_label_uuid)

                    id_gts.append(uid_to_int.get_id(track_label_uuid))
                    num_total_gt += 1

            tracks: Dict[str, Dict[str, Any]] = {}
            id_tracks: List[str] = []

            track_data = read_json_file(path_track_data[ind_frame])

            for track in track_data:
                key = track["track_label_uuid"]

                if track["label_class"] not in cfg.categories or track["height"] == 0:
                    continue

                center = np.array([track["center"]["x"], track["center"]["y"], track["center"]["z"]])
                bbox, orientation = label_to_bbox(track)
                if in_distance_range_pose(np.zeros(3), center, cfg.d_min, cfg.d_max):
                    tracks[key] = {}
                    tracks[key]["centroid"] = center
                    tracks[key]["bbox"] = bbox
                    tracks[key]["orientation"] = orientation
                    tracks[key]["width"] = track["width"]
                    tracks[key]["length"] = track["length"]
                    tracks[key]["height"] = track["height"]

                    id_tracks.append(uid_to_int.get_id(key))

            dists_c: List[List[float]] = []
            dists_i: List[List[float]] = []
            dists_o: List[List[float]] = []
            for gt_key, gt_value in gt.items():
                gt_track_data_c: List[float] = []
                gt_track_data_i: List[float] = []
                gt_track_data_o: List[float] = []
                dists_c.append(gt_track_data_c)
                dists_i.append(gt_track_data_i)
                dists_o.append(gt_track_data_o)
                for track_key, track_value in tracks.items():
                    count_all += 1
                    gt_track_data_c.append(get_distance(gt_value, track_value, "centroid"))
                    gt_track_data_i.append(get_distance(gt_value, track_value, "iou"))
                    gt_track_data_o.append(get_distance(gt_value, track_value, "orientation"))

            assert(np.shape(dists_c) == np.shape(dists_i) and np.shape(dists_c) == np.shape(dists_o))
            # print(np.shape(dists_c), np.shape(dists_i), np.shape(dists_o))
            acc_c.update(id_gts, id_tracks, dists_c)
            acc_i.update(id_gts, id_tracks, dists_i)
            acc_o.update(id_gts, id_tracks, dists_o)

    logger.info(f'{count_all = }')

    logger.info(f'{acc_c = }')

    summary = mh.compute(
        acc_c,
        metrics=[
            "num_frames",
            "mota",
            "motp",
            "idf1",
            "mostly_tracked",
            "mostly_lost",
            "num_false_positives",
            "num_misses",
            "num_switches",
            "num_fragmentations",
        ],
        name="acc",
    )
    logger.info("summary = %s", summary)

    num_tracks = len(ID_gt_all)

    fn = os.path.basename(path_tracker_output)
    num_frames = summary["num_frames"][0]
    mota = summary["mota"][0] * 100
    motp_c = (1 - summary["motp"][0])
    idf1 = summary["idf1"][0]*100
    most_track = summary["mostly_tracked"][0] / num_tracks *100
    most_lost = summary["mostly_lost"][0] / num_tracks *100
    num_fp = summary["num_false_positives"][0]
    num_miss = summary["num_misses"][0]
    num_switch = summary["num_switches"][0]
    num_frag = summary["num_fragmentations"][0]

    acc_c.events.loc[acc_c.events.Type != "RAW", "D"] = acc_i.events.loc[acc_i.events.Type != "RAW", "D"]

    sum_motp_i = mh.compute(acc_c, metrics=["motp"], name="acc")
    logger.info("MOTP-I = %s", sum_motp_i)
    num_tracks = len(ID_gt_all)

    motp_i = sum_motp_i["motp"][0]

    acc_c.events.loc[acc_c.events.Type != "RAW", "D"] = acc_o.events.loc[acc_o.events.Type != "RAW", "D"]
    sum_motp_o = mh.compute(acc_c, metrics=["motp"], name="acc")
    logger.info("MOTP-O = %s", sum_motp_o)
    num_tracks = len(ID_gt_all)

    motp_o = sum_motp_o["motp"][0]

    fn = os.path.basename(path_tracker_output)
    out_df = pd.DataFrame([[fn, num_frames, num_tracks, mota, motp_c, motp_o, motp_i, idf1, most_track, most_lost, num_fp, num_miss, num_switch, num_frag]]
                          , columns=[
                              "File Name",
                              "Num Frames", "Num Tracks", "MOTA", "MOTP_c", "MOTP_o", "MOTP_i",
                              "IDF1", "Most Track", "Most Lost", "Number of False Positive",
                              "Number of Misses", "Number of Switches", "Number of Fragments"])

    out_df.to_csv(os.path.join(cfg.prediction_path, cfg.out_file))

    return {"Categories": list(cfg.categories),
            "Num Frames": num_frames,
            "Num Tracks": num_tracks,
            "MOTA": mota,
            "MOTPC": motp_c,
            "MOTPO": motp_o,
            "MOTPI": motp_i,
            "IDF1": idf1,
            "Most Track": most_track,
            "Most Lost": most_lost,
            "Number of False Positive": num_fp,
            "Number of Misses": num_miss,
            "Number of Switches": num_switch,
            "Number of Fragments": num_frag}
