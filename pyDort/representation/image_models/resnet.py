from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from clort.clearn.data import ArgoverseObjectDataFrame
from torchvision.models import *
from torchvision.models._api import WeightsEnum

from ..simple import SimpleArgoverseDetectionRepresentation


class ResnetImageRepresentation(SimpleArgoverseDetectionRepresentation):

    def __init__(self,
                 model:str = "resnet18",
                 weights:Type[WeightsEnum] = None,
                 gpu:bool = False,
                 agr:str = "max",
                 chunk_size:int = 1) -> None:
        self.avail_models: Dict[str, Callable[..., ResNet]] = \
                                        {
                                        "resnet18":             {"fxn" : resnet18,
                                                                 "weights": ResNet18_Weights.IMAGENET1K_V1}, # test
                                        "resnet34":             {"fxn": resnet34,
                                                                 "weights": ResNet34_Weights.IMAGENET1K_V1},
                                        "resnet50":             {"fxn": resnet50,
                                                                 "weights": ResNet50_Weights.IMAGENET1K_V2}, # test
                                        "resnet101":            {"fxn": resnet101,
                                                                 "weights": ResNet101_Weights.IMAGENET1K_V2},
                                        "resnet152":            {"fxn": resnet152,
                                                                 "weights": ResNet152_Weights.IMAGENET1K_V2}, # test
                                        "resnext50_32x4d":      {"fxn": resnext50_32x4d,
                                                                 "weights": ResNeXt50_32X4D_Weights.IMAGENET1K_V2},
                                        "resnext101_32x8d":     {"fxn": resnext101_32x8d,
                                                                 "weights": ResNeXt101_32X8D_Weights.IMAGENET1K_V2},
                                        "resnext101_64x4d":     {"fxn": resnext101_64x4d,
                                                                 "weights": ResNeXt101_64X4D_Weights.IMAGENET1K_V1},
                                        "wide_resnet50_2":      {"fxn": wide_resnet50_2,
                                                                 "weights": Wide_ResNet50_2_Weights.IMAGENET1K_V2},
                                        "wide_resnet101_2":     {"fxn": wide_resnet101_2,
                                                                 "weights": Wide_ResNet101_2_Weights.IMAGENET1K_V2},
                                        # ConvNext models
                                        "convnext_large":       {"fxn": convnext_large,
                                                                 "weights": ConvNeXt_Large_Weights.IMAGENET1K_V1},
                                        "convnext_small":       {"fxn": convnext_small,
                                                                 "weights": ConvNeXt_Small_Weights.IMAGENET1K_V1}, # test
                                        "convnext_tiny":        {"fxn": convnext_tiny,
                                                                 "weights": ConvNeXt_Tiny_Weights.IMAGENET1K_V1}, # test
                                        "swin_s":               {"fxn": swin_s,
                                                                 "weights": Swin_S_Weights.IMAGENET1K_V1}, # test
                                        "swin_b":               {"fxn": swin_b,
                                                                 "weights": Swin_B_Weights.IMAGENET1K_V1}, # test
                                        "regnet_y_16gf":        {"fxn": regnet_y_16gf,
                                                                 "weights": RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1}, # test
                                        "efficientnet_v2_s":    {"fxn": efficientnet_v2_s,
                                                                 "weights": EfficientNet_V2_S_Weights.IMAGENET1K_V1}, # test
                                        "efficientnet_v2_m":    {"fxn": efficientnet_v2_m,
                                                                 "weights": EfficientNet_V2_M_Weights.IMAGENET1K_V1}, # test
                                        "efficientnet_b3":      {"fxn": efficientnet_b3,
                                                                 "weights": EfficientNet_B3_Weights.IMAGENET1K_V1}, # test
                                        "vit_b16":              {"fxn": vit_b_16,
                                                                 "weights": ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1}, # test
                                        "regnet_x_3_3gf":       {"fxn": regnet_x_3_2gf,
                                                                 "weights": RegNet_X_3_2GF_Weights.IMAGENET1K_V2} # test
                                        }
        assert (model in self.avail_models)

        model_fxn = self.avail_models[model]["fxn"]
        weights = weights if weights is not None else self.avail_models[model]["weights"]

        modules = list(model_fxn(weights=weights).children())[:-1]
        self.device = torch.device("cuda") if torch.cuda.is_available() and gpu else torch.device("cpu")

        self.model = nn.Sequential(*modules)
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
