from pathlib import Path
from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn

from .dgcnn import DGCNN_cls
from .pointnets import PointNetClassifier


class PointCloudRepresentation(nn.Module):

    def __init__(self,
                 model:str = "pointnet",
                 n_points: int = 20,
                 k:int = 10) -> None:
        super().__init__()
        assert(k < n_points)
        self.avail_models: Dict[str, Dict[str, Any]] = \
                                        {
                                        "pointnet":  {"cls": PointNetClassifier,
                                                     "weights": f'{Path(__file__).parent.resolve()}/ckpts/pointnet_model.t7'},
                                        "dgcnn1024": {"cls": DGCNN_cls,
                                                      "weights": f'{Path(__file__).parent.resolve()}/ckpts/DGCNN_model.cls.1024.t7',
                                                      "args": {"k": k,
                                                               "emb_dims": 1024,
                                                               "dropout": 0.5}},
                                        "dgcnn2048": {"cls": DGCNN_cls,
                                                      "weights": f'{Path(__file__).parent.resolve()}/ckpts/DGCNN_model.cls.2048.t7',
                                                      "args": {"k": k,
                                                               "emb_dims": 1024,
                                                               "dropout": 0.5}}
                                        }
        assert (model in self.avail_models)

        self.model: Optional[Type[nn.Module]] = None

        self.out_dim = None

        if model == "pointnet":
            self.model = self.avail_models[model]["cls"]()
            print(f'{self.model.load_state_dict(torch.load(self.avail_models[model]["weights"]), strict=False) = }')
            self.out_dim = 128
        else:
            cls = self.avail_models[model]["cls"]
            args = self.avail_models[model]["args"]
            self.model = cls(args)
            self.model = nn.parallel.DataParallel(self.model)
            print(f'{self.model.load_state_dict(torch.load(self.avail_models[model]["weights"]), strict=False) = }')
            self.model = self.model.module
            self.out_dim = 512

        self.n_ch = 3
        self.n_points = n_points

    def forward(self, pcl: torch.Tensor) -> torch.Tensor:
        # pcl -> [B, c, n_pts]

        enc = self.model(pcl)
        enc = enc/(enc.norm(dim = -1, keepdim=True) + 1e-9)

        return enc
