from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
