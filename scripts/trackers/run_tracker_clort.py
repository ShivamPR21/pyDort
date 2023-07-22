import argparse
import os
from pathlib import Path
from typing import List

import hydra
import numpy as np
import torch
import wandb
from clort.data import ArgoCL
from clort.model import CLModel, DLA34Encoder
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

from pyDort.helpers import UUIDGeneration, check_mkdir, read_json_file, save_json_dict
from pyDort.sem.filterpy_ukf import FilterPyUKF
from pyDort.tracking.pydort import PyDort
from pyDort.tracking.se2 import SE2
from pyDort.tracking.se3 import SE3
from pyDort.tracking.transform_utils import (
    batch_bbox_3d_from_8corners,
    get_B_SE2_A,
    rotmat2d,
    se2_to_yaw,
    yaw_to_quaternion3d,
)


@hydra.main(version_base=None, config_path="../conf", config_name="clort_config")
def run_tracker(cfg: DictConfig) -> None:

    uuid_gen = UUIDGeneration()

    dataset = ArgoCL(cfg.data.data_dir,
                    temporal_horizon=1,
                    temporal_overlap=0,
                    max_objects=cfg.data.max_objects,
                    target_cls=cfg.data.target_cls,
                    distance_threshold=cfg.data.distance_threshold,
                    splits=cfg.data.split,
                    img_size=cfg.data.img_shape,
                    point_cloud_size=cfg.data.pcl_quant,
                    in_global_frame=cfg.data.in_global_frame,
                    pivot_to_first_frame=cfg.data.pivot_to_first_frame,
                    image=cfg.data.image, pcl=cfg.data.pcl, bbox=cfg.data.bbox_aug,
                    vision_transform=None, # type: ignore
                    pcl_transform=None)

    appearance_model = DLA34Encoder(out_dim=256)
    appearance_model = appearance_model.to('cuda')
    appearance_model.eval()

    cur_log = None
    tracker = None

    for i in (run := tqdm(range(dataset.N), total=dataset.N)):
        data = dataset[i]

        rd_i, frame_idx = dataset.get_reduced_index(i) # type: ignore
        log_id : str = dataset.log_files[rd_i].name # type: ignore
        frame_log = dataset.log_files[rd_i][dataset.frames[log_id][frame_idx]]

        tr = np.asanyarray(frame_log['local_to_global_transform'], dtype=np.float32) # type: ignore
        R, t = tr[:, :3], tr[:, 3]
        city_SE3_egovehicle = SE3(R.T, t)
        current_lidar_timestamp = np.asanyarray(frame_log['timestamp'], dtype=np.uint64) # type: ignore

        run.set_description(f'Log Id: {cur_log}')

        pcls, pcls_sz, imgs, imgs_sz, bboxs, track_idxs, cls_idxs, frame_sz = data

        if (len(track_idxs) == 0):
            continue
        track_uids = [dataset.idx_to_tracks_map[log_id][idx] for idx in track_idxs]
        track_cls = (np.array(dataset.obj_cls, dtype=str)[cls_idxs]).tolist()

        # Tracking start
        log_id = log_id.split("_")[-1]
        if log_id != cur_log:
            tracker = PyDort()
            cur_log = log_id

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
        dets_w_info = tracker.update(bboxs.detach().cpu().numpy(), [encoding.detach().cpu().numpy(), None], track_cls)

        tracked_labels = []
        for i, det in enumerate(batch_bbox_3d_from_8corners(bboxs.detach().cpu().numpy())):
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
            qw,qx,qy,qz = yaw_to_quaternion3d(ego_yaw_obj)
            tracked_labels.append({
            "center": {"x": float(xyz_ego[0]), "y": float(xyz_ego[1]), "z": float(xyz_ego[2])},
            "rotation": {"x": float(qx) , "y": float(qy), "z": float(qz) , "w": float(qw)},
            "length": float(det[4]),
            "width": float(det[5]),
            "height": float(det[6]),
            "track_label_uuid": uuid_gen.get_uuid(det[7]+1) if len(det) >= 9 else track_uids[i],
            "timestamp": int(current_lidar_timestamp),
            "label_class": det[8] if len(det) >= 9 else track_cls[i]
            })

        label_dir = os.path.join(cfg.data.tracks_dump_dir, cur_log, "per_sweep_annotations_amodal")
        check_mkdir(label_dir)
        json_fname = f"tracked_object_labels_{current_lidar_timestamp}.json"
        json_fpath = os.path.join(label_dir, json_fname)
        if Path(json_fpath).exists():
            # accumulate tracks of another class together
            prev_tracked_labels = read_json_file(json_fpath)
            tracked_labels.extend(prev_tracked_labels)

        save_json_dict(json_fpath, tracked_labels) # type: ignore

if __name__ == "__main__":
    run_tracker()
