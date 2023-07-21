import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from clort.data import ArgoCL
from clort.model import CLModel, DLA34Encoder
from tqdm import tqdm

from pyDort.helpers import UUIDGeneration, check_mkdir, read_json_file, save_json_dict
from pyDort.sem.data_association import DataAssociation
from pyDort.sem.filterpy_ukf import FilterPyUKF
from pyDort.tracking.pydort import PyDort
from pyDort.tracking.se2 import SE2
from pyDort.tracking.se3 import SE3
from pyDort.tracking.transform_utils import (
    get_B_SE2_A,
    rotmat2d,
    se2_to_yaw,
    yaw_to_quaternion3d,
)


def run_tracker(
    data_dir: str,
    config_file: str,
    tracks_dump_dir: str,
    max_age: int = 3,
    split: List[str] = ["val"],
    min_hits: int = 1,
    min_conf: float = 0.3,
    n_logs: int = 100
    ) -> None:

    uuid_gen = UUIDGeneration()

    dataset = ArgoCL(data_dir,
                    temporal_horizon=1,
                    temporal_overlap=0,
                    max_objects=None,
                    target_cls=None,
                    distance_threshold=None,
                    splits=split,
                    img_size=(128, 128),
                    point_cloud_size=[100, 200],
                    in_global_frame=True,
                    pivot_to_first_frame=False,
                    image=True, pcl=True, bbox=True,
                    vision_transform=None, # type: ignore
                    pcl_transform=None)

    appearance_model = CLModel()
    appearance_model = appearance_model.to('cuda')
    appearance_model.eval()

    da_model = DataAssociation(3e-1, False, True, None, None, np.array([1, 2], dtype=np.float32))

    cur_log = None
    tracker = None

    for i in range(dataset.N):
        data = dataset[i]

        rd_i, frame_idx = dataset.get_reduced_index(i) # type: ignore
        log_id : str = dataset.log_files[rd_i].name # type: ignore
        frame_log = dataset.frames[log_id][frame_idx]

        tr = np.asanyarray(frame_log['local_to_global_transform'], dtype=np.float32) # type: ignore
        R, t = tr[:, :3], tr[:, 3]
        city_SE3_egovehicle = SE3(R.T, t)
        current_lidar_timestamp = np.asanyarray(frame_log['timestamp'], dtype=np.uint64) # type: ignore

        if log_id != cur_log:
            tracker = PyDort(max_age, 1., min_hits, appearance_model, da_model, FilterPyUKF, config_file)

        pcls, pcls_sz, imgs, imgs_sz, bboxs, track_idxs, cls_idxs, frame_sz = data

        if (len(track_idxs) == 0):
            continue

        pcls = pcls.to('cuda') if isinstance(pcls, torch.Tensor) else pcls
        imgs = imgs.to('cuda') if isinstance(imgs, torch.Tensor) else imgs
        bboxs = bboxs.to('cuda') if isinstance(bboxs, torch.Tensor) else bboxs

        mv_e, pc_e, mm_e, mmc_e = appearance_model(pcls, pcls_sz, imgs, imgs_sz, bboxs, frame_sz)
        encoding = None
        if mmc_e is not None:
            encoding = mmc_e
        elif mm_e is not None:
            encoding = mm_e
        elif pc_e is not None:
            encoding = pc_e
        elif mv_e is not None:
            encoding = mv_e
        else:
            raise NotImplementedError("Encoder resolution failed.")

        assert(encoding is not None)
        assert(tracker is not None)
        dets_w_info = tracker.update(bboxs.detach().cpu().numpy(), [encoding.detach().cpu().numpy(), None], (np.array(dataset.obj_cls, dtype=str)[cls_idxs]).tolist())

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

        save_json_dict(json_fpath, tracked_labels) # type: ignore

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
    parser.add_argument("--n_logs", type=int, default=100,
                        help="Number of logs to process.")

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

    args = parser.parse_args()
    print(args)

    run_tracker(
        raw_data_dir=args.raw_data_dir,
        dets_dump_dir=args.dets_dataroot,
        config_file=args.config_file,
        tracks_dump_dir=args.tracks_dump_dir,
        max_age=args.max_age,
        min_hits=args.min_hits,
        min_conf=args.min_conf,
        target_cls=args.target_cls,
        n_logs=args.n_logs
    )
