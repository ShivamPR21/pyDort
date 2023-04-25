""" Copyright (C) 2022  Shiavm Pandey.

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
"""

import argparse
import os
from typing import List

import h5py
import numpy as np
import torch
from tqdm import tqdm

from pyDort.tracking.data_pipe import ArgoverseTrackingInferenceDataset


def dump_data_to_hdf5(
    raw_data_dir: str,
    dets_dump_dir: str,
    tracks_dump_dir: str,
    min_conf: float = 0.3,
    target_cls: List[str] = None,
    n_logs: int = 100,
    data_partition_name: str = "",
    ) -> None:

    dl = ArgoverseTrackingInferenceDataset(raw_data_dir,
                                           dets_dump_dir,
                                           log_id="",
                                           lidar_points_thresh=20,
                                           image_size_threshold=50,
                                           n_img_view_aug=7,
                                           aug_transforms=None,
                                           central_crop=False,
                                           img_tr_ww=(0.95, 0.95),
                                           discard_invalid_dfs=True,
                                           img_reshape=(256, 256),
                                           target_cls=target_cls)
    tracks_dump_dir = os.path.join(tracks_dump_dir, data_partition_name)
    os.makedirs(tracks_dump_dir, exist_ok=True)

    for i, log_id in tqdm(enumerate(dl.log_list[:min(len(dl.log_list), n_logs)])):
        # Init dataset with log_id
        dl.dataset_init(i)

        # Create hdf5 dataset
        file_path_log = os.path.join(tracks_dump_dir, f'log_{log_id}.hdf5')

        log_h5 = h5py.File(file_path_log, 'a', track_order=True)  # open the file in append mode

        log_grp = log_h5.create_group(f'{data_partition_name}_log_{i}_{log_id}', track_order=True)

        for j in range(len(dl)):
            dets = dl[j]
            if dets is None:
                continue

            n_dets = len(dets)
            for k in reversed(range(n_dets)):
                if dets[k].score < min_conf:
                    dets.pop(k)

            print(f'Processing frame:{j+1} in log:{i+1}')

            timestamp = dl.lidar_timestamps[j]
            # Create subgroup for frame id
            frame_h5_grp = log_grp.create_group(f'frame_{j}_{timestamp}', track_order=True)

            # Frame transformation local view -> global view
            local_to_global_transform = np.hstack(
                (dl.city_to_egovehicle_se3_list[j].rotation.T,
                dl.city_to_egovehicle_se3_list[j].translation.reshape((-1, 1))
                ))
            _ = frame_h5_grp.create_dataset(name="local_to_global_transform",
                                              data=local_to_global_transform,
                                              shape=(3, 4),
                                              maxshape=(3, 4),
                                              dtype=np.float32,
                                              compression='gzip',
                                              compression_opts=9)

            for d, detection in enumerate(dets):
                # subgroup for detection
                det_h5_grp = frame_h5_grp.create_group(f'det_{d}', track_order=True)

                # track id
                if detection.track_id is None:
                    detection.track_id = "-"

                _ = det_h5_grp.create_dataset(name="track_id",
                                                data=detection.track_id,
                                                dtype='S50')


                # Detection images insertion
                for v, img in enumerate(detection.img_data[:detection.n_orig_imgs]):
                    if isinstance(img, torch.Tensor):
                        img = img.numpy()

                    img = np.array(img*255., dtype=np.uint8)
                    img = img.transpose(1, 2, 0)

                    _ = det_h5_grp.create_dataset(name=f'img_{v}',
                                                  data=img,
                                                  shape=img.shape,
                                                  maxshape=dl.img_reshape + (3,),
                                                  dtype=np.uint8,
                                                  compression='gzip',
                                                  compression_opts=9)

                # Detection point-cloud insertion
                pcl = np.array(detection.get_lidar(), dtype=np.float32)
                if len(pcl) > 2000:
                    idx_sample = np.random.randint(0, len(pcl), size=2000)
                    pcl = pcl[idx_sample]

                _ = det_h5_grp.create_dataset(name="pcl",
                                            data=pcl,
                                            shape=pcl.shape,
                                            maxshape=(2000, 3),
                                            dtype=np.float32,
                                            compression='gzip',
                                            compression_opts=9)

                # # Global point-cloud
                # pcl = np.array(detection.global_pcl, dtype=np.float32)
                # if len(pcl) > 2000:
                #     idx_sample = np.random.randint(0, len(pcl), size=2000)
                #     pcl = pcl[idx_sample]

                # _ = det_h5_grp.create_dataset(name="global_pcl",
                #                             data=pcl,
                #                             shape=pcl.shape,
                #                             maxshape=(2000, 3),
                #                             dtype=np.float32,
                #                             compression='gzip',
                #                             compression_opts=9)

                # Object type
                if detection.object_type is None:
                    detection.object_type = "-"

                _ = det_h5_grp.create_dataset(name="cls",
                                              data=detection.object_type,
                                              dtype='S20')

                # GroundTruth Bounding Box local
                _ = det_h5_grp.create_dataset(name="bbox",
                                              data=detection.bbox,
                                              shape=(8, 3),
                                              maxshape=(8, 3),
                                              dtype=np.float32,
                                              compression='gzip',
                                              compression_opts=9)

                # _ = det_h5_grp.create_dataset(name="bbox_global",
                #                               data=detection.bbox_global,
                #                               shape=(8, 3),
                #                               maxshape=(8, 3),
                #                               dtype=np.float32,
                #                               compression='gzip',
                #                               compression_opts=9)

            # Timestamp
            _ = frame_h5_grp.create_dataset(name="timestamp",
                                            data=timestamp,
                                            dtype=np.uint64)

        log_h5.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dets_dataroot", type=str,
        required=True, help="path to 3d detections")

    parser.add_argument("--raw_data_dir", type=str,
        required=True, help="path to raw log data (including pose data) for validation or test set")
    parser.add_argument("--n_logs", type=int, default=100,
                        help="Number of logs to process.")

    parser.add_argument("--tracks_dump_dir", type=str,
        default='temp_files',
        help="path to dump generated dataset (as .hdf5 file)")
    parser.add_argument("--min_conf", type=float,
        default=0.3,
        help="minimum allowed confidence for 3d detections to be considered valid")
    parser.add_argument("--target_cls", default=None, action="append",
                        help="Classes to be tracked at once.", required=False)
    parser.add_argument("--d_part", type=str,
        default='train1',
        help="DataSet partition")

    args = parser.parse_args()

    dump_data_to_hdf5(
        raw_data_dir=args.raw_data_dir,
        dets_dump_dir=args.dets_dataroot,
        tracks_dump_dir=args.tracks_dump_dir,
        min_conf=args.min_conf,
        target_cls=args.target_cls,
        n_logs=args.n_logs,
        data_partition_name=args.d_part
    )
