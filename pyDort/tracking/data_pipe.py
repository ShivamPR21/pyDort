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

import os
from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from argoverse.data_loading.object_label_record import (
    ObjectLabelRecord,
    json_label_dict_to_obj_record,
)
from argoverse.data_loading.simple_track_dataloader import (
    SimpleArgoverseTrackingDataLoader,
)
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.calibration import Calibration, load_calib, load_image
from argoverse.utils.camera_stats import RING_CAMERA_LIST
from argoverse.utils.frustum_clipping import generate_frustum_planes
from argoverse.utils.ply_loader import load_ply
from argoverse.utils.se3 import SE3
from clort.clearn.data.dataframe import ArgoverseDataFrame, ArgoverseObjectDataFrame
from clort.clearn.data.utils import get_object_patch_from_image


class ArgoverseTrackingInferenceDataset:

    def __init__(self,
                 data_dir : str,
                 dets_dump_dir : str,
                 log_id : str = "",
                 lidar_points_thresh : int = 30,
                 image_size_threshold : int = 50,
                 n_img_view_aug : int = 7,
                 aug_transforms : List[Callable] = None,
                 central_crop : bool = True,
                 img_tr_ww : Tuple[float, float] = (0.7, 0.7),
                 discard_invalid_dfs : bool = True,
                 img_reshape : Tuple[int, int] = (200, 200)) -> None:

        self.data_dir, self.dets_dump_dir = data_dir, dets_dump_dir
        self.log_id = log_id

        # Validity thresholds
        self.lidar_points_thresh = lidar_points_thresh
        self.image_size_threshold = image_size_threshold

        # Final view augmentation
        self.n_img_view_aug = n_img_view_aug
        self.aug_transforms = aug_transforms

        # Argoverse tracking loader
        self.tracking_loader = \
            SimpleArgoverseTrackingDataLoader(data_dir=self.data_dir,
                                                                 labels_dir=self.dets_dump_dir)
        self.log_list = [log for log in self.tracking_loader.sdb.get_valid_logs()]

        # data processing parameter
        self.central_crop = central_crop
        self.img_tr_ww = img_tr_ww
        self.discard_invalid_dfs = discard_invalid_dfs

        self.img_reshape = img_reshape

        self.am = ArgoverseMap()

        # store data related variables
        self.n_frames = None
        self.labels_folder = None
        self.lidar_timestamps : List[int] = None
        self.city_to_egovehicle_se3_list : List[SE3] = None
        self.calib_data : Dict[Any, Calibration] = None
        self.mdt = -1

        self.curr_frame_data : List[ArgoverseObjectDataFrame] = []

        if self.log_id:
            try:
                self.labels_folder = self.dets_dump_dir + "/" + self.log_id + "/per_sweep_annotations_amodal/"
                lis = os.listdir(self.labels_folder)
                self.lidar_timestamps = [ int(file.split(".")[0].split("_")[-1]) for file in lis]
                self.lidar_timestamps.sort()

                self.mdt = np.array(self.lidar_timestamps, dtype=np.float32)
                self.mdt = (self.mdt[1:] - self.mdt[:-1]).mean()*1e-9 # seconds

                self.n_frames = len(self.lidar_timestamps)
                calib_filename = os.path.join(self.data_dir, self.log_id, "vehicle_calibration_info.json")
                self.calib_data = load_calib(calib_filename)
                self.city_to_egovehicle_se3_list = [None for i in range(self.n_frames)]
            except:
                print(f'The data log_id {self.log_id} not available in dataset\n'
                      'Use `dataset_init` function to specify the dataset log.')

    def dataset_init(self, log_idx : int) -> None:
        self.reset() # Reset dataset
        assert(log_idx < len(self.log_list) and log_idx >= 0)

        self.log_id = self.log_list[log_idx]
        self.labels_folder = self.dets_dump_dir + "/" + self.log_id + "/per_sweep_annotations_amodal/"
        lis = os.listdir(self.labels_folder)
        self.lidar_timestamps = [ int(file.split(".")[0].split("_")[-1]) for file in lis]
        self.lidar_timestamps.sort()

        self.mdt = np.array(self.lidar_timestamps, dtype=np.float32)
        self.mdt = (self.mdt[1:] - self.mdt[:-1]).mean()*1e-9 # seconds

        self.n_frames = len(self.lidar_timestamps)
        calib_filename = os.path.join(self.data_dir, self.log_id, "vehicle_calibration_info.json")
        self.calib_data = load_calib(calib_filename)
        self.city_to_egovehicle_se3_list = [None for i in range(self.n_frames)]

    def reset(self) -> None:
        self.log_id = ""
        self.n_frames = None
        self.labels_folder = None
        self.lidar_timestamps : List[int] = None
        self.city_to_egovehicle_se3_list : List[SE3] = None
        self.calib_data : Dict[Any, Calibration] = None
        self.mdt = -1

    def load_data_frame(self, idx : int):
        self.curr_frame_data : List[ArgoverseObjectDataFrame] = []

        assert(not self.n_frames is None and 0<=idx and idx < self.n_frames)
        timestamp = self.lidar_timestamps[idx]

        self.city_to_egovehicle_se3_list[idx] = \
            self.tracking_loader.get_city_to_egovehicle_se3(self.log_id, timestamp)

        if self.city_to_egovehicle_se3_list[idx] is None:
            return

        # Get point cloud
        pcloud_fpath = self.tracking_loader.get_closest_lidar_fpath(self.log_id, timestamp)
        pcloud = load_ply(pcloud_fpath)

        # Prune point cloud
        pcloud = self.__prune_point_cloud__(idx, pcloud)
        dataframe = ArgoverseDataFrame(timestamp)
        dataframe.set_lidar(pcloud)
        dataframe.set_valid()

        for i, camera in enumerate(RING_CAMERA_LIST):
            img_fpath = \
                self.tracking_loader.get_closest_im_fpath(self.log_id, camera, timestamp)
            img = load_image(img_fpath)

            dataframe.set_iamge(img, i)

        objects = self.tracking_loader.get_labels_at_lidar_timestamp(self.log_id, timestamp)
        for obj in objects:
            obj = json_label_dict_to_obj_record(obj)

            df_obj = ArgoverseObjectDataFrame(timestamp, "-", augmentation=True,
                                              img_resize=self.img_reshape, n_images=self.n_img_view_aug)

            # Store label, and object dimensions
            df_obj.object_type = obj.label_class
            df_obj.dims = np.array([obj.length, obj.width, obj.height], dtype=np.float32)
            df_obj.score = obj.score

            if (self.__populate_object_dataframe__(idx, dataframe, obj, df_obj, self.lidar_points_thresh)):
                df_obj.set_valid()

            if self.discard_invalid_dfs and not df_obj.is_valid():
                continue

            df_obj.generate_inference_img_data()
            self.curr_frame_data.append(df_obj)

    def __populate_object_dataframe__(self,
                                idx : int,
                                df : ArgoverseDataFrame,
                                obj : ObjectLabelRecord,
                                obj_df : ArgoverseObjectDataFrame,
                                cloud_thresh : int = 30) -> bool:
        obj_df.bbox = obj.as_3d_bbox()
        obj_df.bbox_global = self.city_to_egovehicle_se3_list[idx].transform_point_cloud(obj_df.bbox)

        # For lidar dataframe
        pcloud = self.__segment_cloud__(obj_df.bbox, df.lidar)
        if pcloud.shape[0] <= cloud_thresh:
            return False

        pcloud -= pcloud.mean(axis=0, keepdims=True)

        obj_df.set_lidar(pcloud)

        # For camera dataframe
        img_cnt = 0;
        for i, cam in enumerate(RING_CAMERA_LIST):
            calib = self.calib_data[cam]
            img = df.get_image(i)
            planes = generate_frustum_planes(calib.K, calib.camera)
            uv_cam = calib.project_ego_to_cam(obj_df.bbox)
            obj_patch = get_object_patch_from_image(img,
                                                    uv_cam[:, :3],
                                                    planes.copy(),
                                                    deepcopy(calib.camera_config),
                                                    self.central_crop,
                                                    self.img_tr_ww)

            if ((not obj_patch is None) and (obj_patch.shape[0] >= self.image_size_threshold) and (obj_patch.shape[1] >= self.image_size_threshold)):
                img_cnt += 1
                obj_df.set_iamge(obj_patch, i)

        if img_cnt < 1:
            return False

        return True

    def __segment_cloud__(self,
                        box : np.ndarray,
                        lidar : np.ndarray) -> np.ndarray:
        p, p1, p2, p3 = box[0], box[1], box[3], box[4]

        d = lidar - p
        dnorm = np.linalg.norm(d, axis = 1, keepdims=True)
        mask = np.ones((1, len(lidar)), dtype=np.bool8)
        for p_ in [p1, p2, p3]:
            d1 = np.expand_dims(p_ - p, axis = 0)
            d1norm = np.linalg.norm(d1)
            cost = d1.dot(d.T)/(dnorm.T*d1norm)
            dist = dnorm.T*cost
            tmp_mask = np.logical_and(dist >= 0, dist <= d1norm)
            mask = np.logical_and(mask, tmp_mask)

        return lidar[mask[0]]

    def __prune_point_cloud__(self,
                            idx : int,
                            lidar : np.ndarray,
                            prune_non_roi : bool = True,
                            prune_ground : bool = True) -> np.ndarray:
        city_to_egovehicle_se3 = self.city_to_egovehicle_se3_list[idx]
        assert(city_to_egovehicle_se3 is not None)

        city_name = self.tracking_loader.get_city_name(self.log_id)

        roi_area_pts = deepcopy(lidar)
        roi_area_pts = city_to_egovehicle_se3.transform_point_cloud(
            roi_area_pts
        )
        if prune_non_roi:
            roi_area_pts = self.am.remove_non_roi_points(roi_area_pts, city_name)

        if prune_ground:
            roi_area_pts = self.am.remove_ground_surface(roi_area_pts, city_name)

        roi_area_pts = city_to_egovehicle_se3.inverse_transform_point_cloud(
            roi_area_pts
        )
        return roi_area_pts

    def __getitem__(self, idx:int) -> List[ArgoverseObjectDataFrame]:
        self.load_data_frame(idx)
        return self.curr_frame_data

    def __len__(self) -> int:
        return self.n_frames
