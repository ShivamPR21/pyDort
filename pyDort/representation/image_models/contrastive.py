from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from clort.clearn.data import ArgoverseObjectDataFrame

from ..simple import SimpleArgoverseDetectionRepresentation


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim=10):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18,
                            "resnet50": models.resnet50}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features
        print(dim_mlp)

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, 128))

    def _get_basemodel(self, model_name):
        model = self.resnet_dict[model_name]()
        return model

    def forward(self, x):
        return self.backbone(x)

class ResnetSimCLRInference(nn.Module):

    def __init__(self, base_model = "resnet18") -> None:
        super().__init__()
        self.weigths = {"resnet18": f'{Path(__file__).parent.resolve()}/chkpts/resnet18_cifar10_cl.tar',
                        "resnet50": f'{Path(__file__).parent.resolve()}/chkpts/resnet50_stl10_cl.pth.tar'}
        self.model = ResNetSimCLR(base_model)
        self.model.load_state_dict(torch.load(self.weigths[base_model], map_location="cuda:0")["state_dict"])

        self.model.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)


class ImageContrastiveRepresentation(SimpleArgoverseDetectionRepresentation):

    def __init__(self,
                 model:str = "resnet18",
                 gpu:bool = False,
                 agr:str = "max",
                 chunk_size:int = 1) -> None:

        self.device = torch.device("cuda") if torch.cuda.is_available() and gpu else torch.device("cpu")

        self.model = ResnetSimCLRInference(model)
        self.model.to(self.device)
        self.model.eval()

        self.n_ch = 3
        self.agr = agr
        self.chunk_size = chunk_size

    def __call__(self, data: List[ArgoverseObjectDataFrame], state_aug: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if len(data) == 0:
            tmp = np.empty((0, 1), dtype=np.float32)
            return (tmp, tmp) if state_aug else (tmp, None)

        features = []
        n_obj = len(data)
        for idx in range(0, n_obj, self.chunk_size):
            l_idx, r_idx = idx, idx+min(n_obj, idx+self.chunk_size)
            feature_chunk = self.feature_extractor(*self.data_prep(data[l_idx:r_idx]))
            features += [feature_chunk]
        features = np.concatenate(features, axis=0)

        if state_aug:
            states = [self.extract_state(obj) for obj in data]
            states = np.stack(states, axis=0)

            return features, states

        return features, None

    @staticmethod
    def data_prep(data: List[ArgoverseObjectDataFrame]) -> Tuple[torch.Tensor, int, int]:
        n_images = data[0].n_images
        batch_size = len(data)
        data_ = []

        for obj in data:
            imgs = torch.cat([img.unsqueeze(dim=0) for img in obj.img_data], dim=0) # [n_imgs, 3, H, W]
            data_ += [imgs]

        data_ = torch.cat(data_, dim=0) # [batch_size*n_imgs, 3, H, W]

        return data_, n_images, batch_size

    def feature_extractor(self, data:torch.Tensor, n_images:int, batch_size:int) -> np.ndarray:
        data = data.to(self.device)

        # print(f'Got shapes: {data.shape}, n_images : {n_images}, batch_size : {batch_size}')
        # raise NotImplementedError TODO:Remove excess code

        features: torch.Tensor = self.model(data)
        features = features.flatten(1).unsqueeze(1)

        features = torch.cat(features.split(batch_size), dim=1) # [batch_size, n_images, D]
        assert(features.size()[[0] == batch_size and features.size()[1] == n_images])

        if self.agr == "max":
            features, _ = features.max(dim=1)
        elif self.agr in ["avg", "mean"]:
            features = features.mean(dim=1)
        elif self.agr == "median":
            features, _ = features.median(dim=1)
        elif self.agr == "mode":
            features, _ = features.mode(dim=1)
        else:
            raise NotImplementedError

        assert (features.ndim == 2)
        features = features.detach().cpu().numpy()

        return features
