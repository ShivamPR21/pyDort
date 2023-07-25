import os
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import torch
import wandb
from clort.data import ArgoCL
from clort.model import CLModel, DLA34Encoder
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm

from pyDort.helpers import (
    UUIDGeneration,
    check_mkdir,
    flatten_cfg,
    read_json_file,
    save_json_dict,
)
from pyDort.sem.filterpy_ukf import FilterPyUKF
from pyDort.sem.instance import InstanceSEM
from pyDort.tracking.eval import eval_tracks
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
    wandb.login()

    wandb.init(
        project=cfg.wb.project,

        config=flatten_cfg(cfg)
    )

    uuid_gen = UUIDGeneration()

    sem_module = None
    if cfg.tracker.sem == "ukf":
        sem_module = FilterPyUKF
    elif cfg.tracker.sem == "instance":
        sem_module = InstanceSEM
    else:
        raise NotImplementedError

    dataset = ArgoCL(cfg.data.data_dir,
                    temporal_horizon=1,
                    temporal_overlap=0,
                    max_objects=cfg.data.max_objects,
                    target_cls=cfg.data.target_cls,
                    detection_score_threshold=cfg.data.det_score,
                    splits=list(cfg.data.split),
                    distance_threshold=cfg.data.distance_threshold,
                    img_size=tuple(cfg.data.img_shape),
                    point_cloud_size=list(cfg.data.pcl_quant),
                    in_global_frame=cfg.data.global_frame,
                    pivot_to_first_frame=cfg.data.pivot_to_first_frame,
                    image=cfg.data.imgs, pcl=cfg.data.pcl, bbox=cfg.data.bbox_aug,
                    vision_transform=None, # type: ignore
                    pcl_transform=None,
                    random_miss=cfg.data.random_miss)

    appearance_model = CLModel(mv_backbone=cfg.am.mv_backbone,
                               mv_features=cfg.am.mv_features,
                               mv_xo=cfg.am.mv_xo,
                               pc_features=cfg.am.pc_features,
                               bbox_aug=cfg.am.bbox_aug,
                               pc_xo=cfg.am.pc_xo,
                               mm_features=cfg.am.mm_features,
                               mm_xo=cfg.am.mm_xo,
                               mmc_features=cfg.am.mmc_features)
    ckpt = torch.load(wandb.restore(name=cfg.am.model_file, run_path=cfg.am.run_path, replace=True).name)
    print(f'{appearance_model.load_state_dict(ckpt["enc"]) = }')
    model_device = cfg.am.device

    appearance_model = appearance_model.to(model_device)
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

        pcls, pcls_sz, imgs, imgs_sz, bboxs, track_idxs, cls_idxs, frame_sz = data

        # if (len(track_idxs) == 0):
        #     continue

        track_cls = (np.array(dataset.obj_cls, dtype=str)[cls_idxs]).tolist()

        # Tracking start
        log_id = log_id.split("_")[-1]
        if log_id != cur_log:
            tracker = PyDort(max_age=cfg.tracker.max_age,
                             dt=1.,
                             min_hits=cfg.tracker.min_hits,
                             sem=sem_module,
                             config_file=cfg.tracker.sem_cfg,
                             rep_update=cfg.tracker.rep_update,
                             Q=cfg.tracker.Q,
                             alpha_thresh=cfg.tracker.alpha_t,
                             beta_thresh=cfg.tracker.beta_t,
                             state_w=cfg.tracker.state_w,
                             dsc_w=cfg.tracker.dsc_w,
                             cm_fusion_w=cfg.tracker.cm_fusion_w,
                             trks_center_w=cfg.tracker.track_center_momentum,
                             matching_threshold=cfg.tracker.matching_threshold,
                             favourable_weight=cfg.tracker.fav_w)
            cur_log = log_id

        run.set_description(f'Log Id: {cur_log}')

        encoding = None
        if (len(track_idxs) != 0):
            pcls = pcls.to(model_device) if isinstance(pcls, torch.Tensor) else pcls
            imgs = imgs.to(model_device) if isinstance(imgs, torch.Tensor) else imgs
            bboxs = bboxs.to(model_device) if isinstance(bboxs, torch.Tensor) else bboxs

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

            encoding = encoding.detach().cpu().numpy() # go to cpu for encoding
            bboxs = bboxs.detach().cpu().numpy() # go to numpy for bounding boxes
        else:
            encoding = np.empty((0, 1))
            bboxs = np.empty((0, 8, 3))


        # assert(encoding is not None)
        assert(tracker is not None)
        dets_w_info = tracker.update(bboxs, [encoding, None], track_cls)

        tracked_labels = []
        for _i, det in enumerate(dets_w_info):
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
            "track_label_uuid": uuid_gen.get_uuid(det[7]+1),
            "timestamp": int(current_lidar_timestamp),
            "label_class": det[8]
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

    if cfg.eval.eval:
        ## Evaluate results
        evaluate(cfg)

def evaluate(cfg: DictConfig):
    ## Evaluate results
    dct_eval = OmegaConf.to_container(cfg.eval, resolve=True)
    dct_eval.update({"prediction_path": cfg.data.tracks_dump_dir}) # type: ignore

    for i, categories in enumerate(["categories_full", "categories_vru", "categories_conventional"]): # type: ignore
        dct_eval.update({"categories": cfg.eval[categories]}) # type: ignore
        dct_eval.update({"out_file": f'{categories}_{cfg.eval.out_file}'}) # type: ignore
        dct_eval = OmegaConf.create(dct_eval)
        res = eval_tracks(dct_eval) # type: ignore
        res.update({"Eval Index": i+1})
        wandb.log(res) # type: ignore
        wandb.save(os.path.join(cfg.data.tracks_dump_dir, dct_eval.out_file))

if __name__ == "__main__":
    run_tracker()
