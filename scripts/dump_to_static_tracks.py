import argparse
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from argoverse.data_loading.simple_track_dataloader import (
    SimpleArgoverseTrackingDataLoader,
)
from pyDort.helpers import read_json_file

CLASSES = [
    "PEDESTRIAN",
    "VEHICLE"
]

TANGO_COLORS = np.array(
    [
    [252, 233, 79],
    [237, 212, 0],
    [196, 160, 0],
    [252, 175, 62],
    [245, 121, 0],
    [206, 92, 0],
    [233, 185, 110],
    [193, 125, 17],
    [143, 89, 2],
    [138, 226, 52],
    [115, 210, 22],
    [78, 154, 6],
    [114, 159, 207],
    [52, 101, 164],
    [32, 74, 135],
    [173, 127, 168],
    [117, 80, 123],
    [92, 53, 102],
    [239, 41, 41],
    [204, 0, 0],
    [164, 0, 0],
    [238, 238, 236],
    [211, 215, 207],
    [186, 189, 182],
    [136, 138, 133],
    [85, 87, 83],
    [46, 52, 54],
    [252, 233, 79],
    [237, 212, 0],
    [196, 160, 0],
    [252, 175, 62],
    [245, 121, 0],
    [206, 92, 0],
    [233, 185, 110],
    [193, 125, 17],
    [143, 89, 2],
    [138, 226, 52],
    [115, 210, 22],
    [78, 154, 6],
    [114, 159, 207],
    [52, 101, 164],
    [32, 74, 135],
    [173, 127, 168],
    [117, 80, 123],
    [92, 53, 102],
    [239, 41, 41],
    [204, 0, 0],
    [164, 0, 0]
], dtype=np.uint8)

GRAY_SCALE = np.array([
    [220, 220, 220],
    [211, 211, 211],
    [192, 192, 192],
    [169,169,169],
    [128,128,128],
    [105,105,105],
    [119,136,153],
    [112,128,144],
    [47,79,79]
    ], dtype=np.uint8)

def update_tracks_list(labels:List[Dict[str, Any]],
                       tracks:Dict[str, Any],
                       transform:Any):
    coc = np.zeros((1, 3), dtype=np.float32)
    coc = transform.transform_point_cloud(coc)
    for label in labels:
        c = label['center']
        c = np.array([[c['x'], c['y'], c['z']]], dtype=np.float32)
        c = transform.transform_point_cloud(c)

        if np.linalg.norm(c-coc) > 70:
            continue

        track_id = label['track_label_uuid']
        track_class = label['label_class']

        if track_id in tracks:
            tracks[track_id][1].append(c)
        else:
            if track_class in CLASSES:
                tracks.update({track_id: [track_class, [c]]})

def prepare_tracks(gt_log_path:str,
                   pred_log_path:str,
                   loader:SimpleArgoverseTrackingDataLoader,
                   log:str,
                   time_stamps:List[int]) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]],
                                                   Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
    gt_tracks: Dict[str, List[np.ndarray]] = {}
    # gt_vh_tracks: Dict[str, List[np.ndarray]] = {}
    pred_tracks: Dict[str, List[np.ndarray]] = {}
    # pred_vh_tracks: Dict[str, List[np.ndarray]] = {}

    for time in time_stamps:
        city_to_egovehicle_se3 = loader.get_city_to_egovehicle_se3(log, time)
        if city_to_egovehicle_se3 is None:
            continue

        f_name = "tracked_object_labels_" + str(time) + ".json"
        gt_file = os.path.join(gt_log_path, f_name)
        pred_file = os.path.join(pred_log_path, f_name)

        gt_labels, pred_labels = read_json_file(gt_file), read_json_file(pred_file)

        update_tracks_list(gt_labels, gt_tracks, city_to_egovehicle_se3)
        update_tracks_list(pred_labels, pred_tracks, city_to_egovehicle_se3)

    return gt_tracks, pred_tracks

def create_geometries(tracks:Dict[str, List[np.ndarray]], colors_list:np.ndarray, z_shift:int = 0) -> pd.DataFrame:
    data = np.empty((0, 7), dtype=np.float32)

    for id, [label, states] in tracks.items():
        colors = [colors_list[np.random.choice(len(colors_list), 1)]]*len(states)
        labels = [np.array([label])]*len(states)
        colors = np.vstack(colors)
        state = np.vstack(states)
        state[:, 2] += z_shift
        labels = np.vstack(labels)
        data = np.append(data, np.concatenate((state, colors, labels), axis=1), axis=0)

    df = pd.DataFrame(data, columns=['x', 'y', 'z', 'R', 'G', 'B', 'Label Class'])

    return df

def dump_to_static_tracks(gt_tracks_dir: str,
                          pred_tracks_dir: str):
    tracking_loader = SimpleArgoverseTrackingDataLoader(data_dir=gt_tracks_dir,
                                                        labels_dir=gt_tracks_dir)
    logs = os.listdir(gt_tracks_dir)
    for log in logs:
        print(log)
        gt_log_path = os.path.join(gt_tracks_dir, log, "per_sweep_annotations_amodal") # GT: Log directory path
        pred_log_path = os.path.join(pred_tracks_dir, log, "per_sweep_annotations_amodal") # Pred: Log directory path

        if not (os.path.exists(gt_log_path) and os.path.exists(pred_log_path)):
            continue

        time_stamps = [int((file.split('_')[-1]).split('.')[0]) for file in os.listdir(gt_log_path)]
        time_stamps.sort()

        gt_tracks, pred_tracks = \
            prepare_tracks(gt_log_path,
                           pred_log_path,
                           tracking_loader,
                           log, time_stamps)

        gt_df = create_geometries(gt_tracks, GRAY_SCALE, 0)
        for cl in CLASSES:
            loc_df = gt_df[gt_df.loc[:, 'Label Class'] == cl]
            loc_df.to_csv(f'{pred_log_path}/True_{cl}_tracks.csv', index=False)

        pred_df = create_geometries(pred_tracks, TANGO_COLORS, 10)
        for cl in CLASSES:
            loc_df = pred_df[pred_df.loc[:, 'Label Class'] == cl]
            loc_df.to_csv(f'{pred_log_path}/Pred_{cl}_tracks.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("gt_tracks_dir")
    parser.add_argument("pred_tracks_dir")

    args = parser.parse_args()

    for method in os.listdir(args.pred_tracks_dir):
        method_dir = os.path.join(args.pred_tracks_dir, method)
        if not os.path.isdir(method_dir): continue

        for part in os.listdir(method_dir):
            part_path = os.path.join(method_dir, part)
            if not os.path.isdir(part_path): continue

            for method_v in os.listdir(part_path):
                method_v_dir = os.path.join(part_path, method_v)

                if not os.path.isdir(method_v_dir): continue

                dump_to_static_tracks(args.gt_tracks_dir, method_v_dir)

    print("Done!")
