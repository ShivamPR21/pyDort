from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim=10):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18,
                            "resnet50": models.resnet50}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

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
        print(f'{self.model.load_state_dict(torch.load(self.weigths[base_model])["state_dict"], strict=False) = }')

        self.model.layer4 = nn.Identity()
        self.model.backbone.fc = nn.Identity()

        self.out_dim = 512 if base_model == 'resnet18' else 2048

    def forward(self, x):
        return self.model(x)


class ImageContrastiveRepresentation(ResnetSimCLRInference):

    def __init__(self, base_model="resnet18") -> None:
        super().__init__(base_model)

    def forward(self, imgs: torch.Tensor, imgs_sz: np.ndarray) -> torch.Tensor:
        enc = self.model(imgs).flatten(1)
        enc = torch.cat([spl.max(dim=0, keepdim=True).values for spl in enc.split(imgs_sz.tolist(), dim=0)], dim=0)
        enc = enc/(enc.norm(dim=1, keepdim=True)+1e-9)
        return enc
