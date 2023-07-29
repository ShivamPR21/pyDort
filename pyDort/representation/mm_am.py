import torch
import torch.nn as nn

from .image_models import (
    DetectionRepresentation,
    ImageContrastiveRepresentation,
    ReIdRepresentation,
    ResnetImageRepresentation,
)
from .point_cloud_models import PointCloudRepresentation


class MultiModalEncoder(nn.Module):

    def __init__(self, im_model: str | None, pc_model: str | None) -> None:
        super().__init__()

        self.im_model, self.im_out_dim = None, None
        self.pc_model, self.pc_out_dim = None, None

        if im_model is not None:
            if im_model.startswith('cl'):
                im_model = im_model.split('.')[-1]
                self.im_model = ImageContrastiveRepresentation(im_model)
                print(f'Using Contrastive Model: {im_model}')
            elif im_model.startswith('res'):
                im_model = im_model.split('.')[-1]
                self.im_model = ResnetImageRepresentation(im_model)
                print(f'Using Residual model: {im_model}')
            elif im_model.startswith('reid'):
                im_model = im_model.split('.')[-1]
                self.im_model = ReIdRepresentation(im_model)
                print(f'Using ReId model: {im_model}')
            elif im_model.startswith('det'):
                im_model = im_model.split('.')[-1]
                self.im_model = DetectionRepresentation(im_model)
                print(f'Using ReId model: {im_model}')
            else:
                raise NotImplementedError(f'Image model not implemented : {im_model}')

            self.im_out_dim = self.im_model.out_dim

        if pc_model is not None:
            self.pc_model = PointCloudRepresentation(pc_model, n_points=200, k = 10)

            self.pc_out_dim = self.pc_model.out_dim

    def forward(self, imgs: torch.Tensor, imgs_sz: torch.Tensor, pcl: torch.Tensor) -> torch.Tensor:

        img_enc = self.im_model(imgs, imgs_sz) if self.im_model is not None else None
        pcl_enc = self.pc_model(pcl) if self.pc_model is not None else None

        img_enc = img_enc/(img_enc.norm(dim=-1, keepdim=True)+1e-9) if img_enc is not None else None
        pcl_enc = pcl_enc/(pcl_enc.norm(dim=-1, keepdim=True)+1e-9) if pcl_enc is not None else None

        return img_enc, pcl_enc
