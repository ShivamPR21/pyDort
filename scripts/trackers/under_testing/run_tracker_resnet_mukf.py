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

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
from argoverse.utils.se2 import SE2
from pyDort.helpers import UUIDGeneration, check_mkdir, read_json_file, save_json_dict
from pyDort.representation import ResnetImageRepresentation
from pyDort.sem.data_association import DataAssociation
from pyDort.sem.ukf import UKF
from pyDort.tracking.data_pipe import ArgoverseTrackingInferenceDataset
from pyDort.tracking.pydort import PyDort
from pyDort.tracking.transform_utils import (
    get_B_SE2_A,
    rotmat2d,
    se2_to_yaw,
    yaw_to_quaternion3d,
)
from tqdm import tqdm


def run_tracker(
    raw_data_dir: str,
    dets_dump_dir: str,
    config_file: str,
    tracks_dump_dir: str,
    max_age: int = 3,
    min_hits: int = 1,
    min_conf: float = 0.3,
    target_cls: List[str] = ["PEDESTRIAN", "VEHICLE"],
    model: str = "resnet18",
    gpu: bool = False,
    agr: str = "max",
    chunk_size: int = 1
    ) -> None:

    uuid_gen = UUIDGeneration()

    dl = ArgoverseTrackingInferenceDataset(raw_data_dir,
                                           dets_dump_dir,
                                           log_id="",
                                           lidar_points_thresh=10,
                                           image_size_threshold=50,
                                           n_img_view_aug=7,
                                           aug_transforms=None,
                                           central_crop=True,
                                           img_tr_ww=(0.9, 0.9),
                                           discard_invalid_dfs=True,
                                           img_reshape=(128, 128),
                                           target_cls=target_cls)

    appearance_model = ResnetImageRepresentation(model, None, gpu, agr, chunk_size)
    da_model = DataAssociation(9e-2, False, True, None, None, np.array([2, 1], dtype=np.float32))

    for i, log_id in tqdm(enumerate(dl.log_list[:1])):
        # Init dataset with log_id
        dl.dataset_init(i)
        tracker = PyDort(max_age, dl.mdt, min_hits, appearance_model, da_model, UKF, config_file)

        for j in range(len(dl)):
            dets = dl[j]
            if dets is None: continue

            city_SE3_egovehicle, current_lidar_timestamp = dl.city_to_egovehicle_se3_list[j], dl.lidar_timestamps[j]

            n_dets = len(dets)
            for i in reversed(range(n_dets)):
                if dets[i].score < min_conf:
                    dets.pop(i)

            dets_w_info = tracker.update(dets)

            tracked_labels = []
            for det in dets_w_info:
                # move city frame tracks back to ego-vehicle frame
                xyz_city = np.array([det[0], det[1], det[2]]).reshape(1,3)
                city_yaw_object = det[3]
                city_se2_object = SE2(rotation=rotmat2d(city_yaw_object), translation=xyz_city.squeeze()[:2])
                city_se2_egovehicle, city_yaw_ego = get_B_SE2_A(city_SE3_egovehicle)
                ego_se2_city = city_se2_egovehicle.inverse()
                egovehicle_se2_object = ego_se2_city.right_multiply_with_se2(city_se2_object)

                # recreate all 8 points
                # transform them
                # compute yaw from 8 points once more
                egovehicle_SE3_city = city_SE3_egovehicle.inverse()
                xyz_ego = egovehicle_SE3_city.transform_point_cloud(xyz_city).squeeze()
                # update for new yaw
                # transform all 8 points at once, then compute yaw on the fly

                ego_yaw_obj = se2_to_yaw(egovehicle_se2_object)
                qx,qy,qz,qw = yaw_to_quaternion3d(ego_yaw_obj)
                tracked_labels.append({
                "center": {"x": xyz_ego[0], "y": xyz_ego[1], "z": xyz_ego[2]},
                "rotation": {"x": qx , "y": qy, "z": qz , "w": qw},
                "length": det[4],
                "width": det[5],
                "height": det[6],
                "track_label_uuid": uuid_gen.get_uuid(det[7]+1),
                "timestamp": current_lidar_timestamp,
                "label_class": det[8]
                })

            label_dir = os.path.join(tracks_dump_dir, log_id, "per_sweep_annotations_amodal")
            check_mkdir(label_dir)
            json_fname = f"tracked_object_labels_{current_lidar_timestamp}.json"
            json_fpath = os.path.join(label_dir, json_fname)
            if Path(json_fpath).exists():
                # accumulate tracks of another class together
                prev_tracked_labels = read_json_file(json_fpath)
                tracked_labels.extend(prev_tracked_labels)

            save_json_dict(json_fpath, tracked_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_age", type=int, default=15,
            help="max allowed track age since last measurement update")
    parser.add_argument("--min_hits", type=int, default=1,
        help="minimum number of required hits for track birth")

    parser.add_argument("--dets_dataroot", type=str,
        required=True, help="path to 3d detections")

    parser.add_argument("--raw_data_dir", type=str,
        required=True, help="path to raw log data (including pose data) for validation or test set")

    parser.add_argument("--tracks_dump_dir", type=str,
        default='temp_files',
        help="path to dump generated tracks (as .json files)")
    parser.add_argument("--min_conf", type=float,
        default=0.3,
        help="minimum allowed confidence for 3d detections to be considered valid")
    parser.add_argument("--target_cls", action="append", help="Classes to be tracked at once.", required=True)
    parser.add_argument("--config_file", type=str,
        default=f'{Path(__file__).parent.resolve()}/../pyDort/tracking/conf.json',
        help="Config file, containing information about different parameters.")
    parser.add_argument("--model", type=str,
                        default="resnet18", help="Image model to be used for encodings.")
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction,
                        help="If given and available gpu will be used.")
    parser.add_argument("--agr", type=str, default="max",
                        help="one of the : max, avg, median, mode")
    parser.add_argument("--chunk_size", type=int, default=10,
                        help="Model chunk size")

    args = parser.parse_args()

    run_tracker(
        raw_data_dir=args.raw_data_dir,
        dets_dump_dir=args.dets_dataroot,
        config_file=args.config_file,
        tracks_dump_dir=args.tracks_dump_dir,
        max_age=args.max_age,
        min_hits=args.min_hits,
        min_conf=args.min_conf,
        target_cls=args.target_cls,
        model=args.model,
        gpu=args.gpu,
        agr=args.agr,
        chunk_size=args.chunk_size,
    )
