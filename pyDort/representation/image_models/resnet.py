from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as vm
from torchvision.models._api import WeightsEnum

__ModelType = Optional[Union[vm.ResNet, vm.ConvNeXt, vm.SwinTransformer, vm.RegNet, vm.EfficientNet, vm.VisionTransformer]]

class ResnetImageRepresentation(nn.Module):

    def __init__(self, model:str = "resnet18",
                 weights:Type[WeightsEnum] = None) -> None:
        super().__init__()
        self.avail_models: Dict[str, Any] = \
            {

            "resnet18":             {"fxn" : vm.resnet18,
                                        "weights": vm.ResNet18_Weights.IMAGENET1K_V1,
                                        "out_dim": 256},
            "resnet50":             {"fxn": vm.resnet50,
                                        "weights": vm.ResNet50_Weights.IMAGENET1K_V2,
                                        "out_dim": 1024}, # test
            # ConvNext models
            "convnext_large":       {"fxn": vm.convnext_large,
                                        "weights": vm.ConvNeXt_Large_Weights.IMAGENET1K_V1,
                                        "out_dim": 1536}, # test
            "convnext_small":       {"fxn": vm.convnext_small,
                                        "weights": vm.ConvNeXt_Small_Weights.IMAGENET1K_V1,
                                        "out_dim": 768},
            }
        assert (model in self.avail_models)

        model_fxn = self.avail_models[model]["fxn"]

        self.out_dim = self.avail_models[model]["out_dim"]

        weights = weights if weights is not None else self.avail_models[model]["weights"]

        self.model : __ModelType = None
        self.model = model_fxn(weights=weights)

        if "res" in model:
            self.model.layer4 = nn.Identity()
            self.model.fc = nn.Identity()
        elif "convnext" in model:
            # self.model.features = nn.Sequential(
            #     *(
            #         list(self.model.features.children())[:-1]
            #         )
            #     )
            self.model.classifier = nn.Identity()
        elif "swin" in model:
            self.model.head = nn.Identity()
        elif "reg" in model:
            self.model.fc = nn.Identity()
        elif "efficient" in model:
            self.model.classifier = nn.Identity()
        elif "vit" in model:
            self.model.heads = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, imgs: torch.Tensor, imgs_sz: np.ndarray) -> torch.Tensor:
        enc = self.model(imgs).flatten(1)

        enc = torch.cat([spl.max(dim=0, keepdim=True).values for spl in enc.split(imgs_sz.tolist(), dim=0)], dim=0)
        enc = enc/(enc.norm(dim=1, keepdim=True)+1e-9)
        return enc
