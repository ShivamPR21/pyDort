from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from clort.clearn.data.dataframe import ArgoverseObjectDataFrame

from ..simple import SimpleArgoverseDetectionRepresentation
from .dgcnn import DGCNN_cls
from .pointnets import PointNetClassifier


class PointCloudRepresentation(SimpleArgoverseDetectionRepresentation):

    def __init__(self,
                 model:str = "pointnet",
                 gpu:bool = False,
                 chunk_size:int = 1,
                 n_points: int = 20,
                 k:int = 10) -> None:
        assert(k < n_points)
        self.avail_models: Dict[str, Dict[str, Any]] = \
                                        {
                                        "pointnet":  {"cls": PointNetClassifier,
                                                     "weights": f'{Path(__file__).parent.resolve()}/ckpts/pointnet_model.t7'},
                                        "dgcnn1024": {"cls": DGCNN_cls,
                                                      "weights": f'{Path(__file__).parent.resolve()}/ckpts/DGCNN_model.cls.1024.t7',
                                                      "args": {"k": k,
                                                               "emb_dims": 1024,
                                                               "dropout": 0.5,
                                                               "gpu": gpu}},
                                        "dgcnn2048": {"cls": DGCNN_cls,
                                                      "weights": f'{Path(__file__).parent.resolve()}/ckpts/DGCNN_model.cls.2048.t7',
                                                      "args": {"k": k,
                                                               "emb_dims": 1024,
                                                               "dropout": 0.5,
                                                               "gpu": gpu}}
                                        }
        assert (model in self.avail_models)

        self.model: Optional[Type[nn.Module]] = None

        if model == "pointnet":
            self.model = self.avail_models[model]["cls"]()
            self.model.load_state_dict(torch.load(self.avail_models[model]["weights"]))
        else:
            cls = self.avail_models[model]["cls"]
            args = self.avail_models[model]["args"]
            self.model = cls(args)
            self.model = nn.parallel.DataParallel(self.model)
            self.model.load_state_dict(torch.load(self.avail_models[model]["weights"]))
            self.model = self.model.module


        self.device = torch.device("cuda") if torch.cuda.is_available() and gpu else torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

        self.n_ch = 3
        self.chunk_size = chunk_size
        self.n_points = n_points

    def __call__(self, data: List[ArgoverseObjectDataFrame], state_aug: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if len(data) == 0:
            tmp = np.empty((0, 1), dtype=np.float32)
            return (tmp, tmp) if state_aug else tmp, None

        features = []
        n_obj = len(data)
        for idx in range(0, n_obj, self.chunk_size):
            l_idx, r_idx = idx, idx+min(n_obj, idx+self.chunk_size)
            feature_chunk = self.feature_extractor(self.data_prep(data[l_idx:r_idx], self.n_points))
            features += [feature_chunk]
        features = np.concatenate(features, axis=0)

        if state_aug:
            states = [self.extract_state(obj) for obj in data]
            states = np.stack(states, axis=0)

            return features, states

        return features, None

    @staticmethod
    def data_prep(data: List[ArgoverseObjectDataFrame], n_points) -> Tuple[torch.Tensor, int, int]:
        batch_size = len(data)
        data_ = []

        for obj in data:
            lidar = obj.lidar
            sel_idx = np.random.choice(lidar.shape[0], n_points, replace=lidar.shape[0] < n_points)
            n_pts = torch.tensor(lidar[sel_idx, :], dtype=torch.float32)
            data_ += [n_pts.unsqueeze(0).permute((0, 2, 1))]

        data_ = torch.cat(data_, dim=0) # [batch_size, 3, n_points]

        return data_

    def feature_extractor(self, data:torch.Tensor) -> np.ndarray:
        data = data.to(self.device)

        # print(f'Got shapes: {data.shape}, n_images : {n_images}, batch_size : {batch_size}')
        # raise NotImplementedError TODO:Remove excess code

        features: torch.Tensor = self.model(data)
        features = features.detach().numpy()
        return features
