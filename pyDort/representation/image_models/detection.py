from typing import Any, Dict, Type

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as vm
from torchvision.models._api import WeightsEnum


class DetectionRepresentation(nn.Module):

    def __init__(self, model:str = "FCOS_resnet50",
                 weights:Type[WeightsEnum] = None) -> None:
        super().__init__()
        self.avail_models: Dict[str, Any] = \
            {
                "FCOS_resnet50": {'fxn': vm.detection.fcos_resnet50_fpn,
                                  'weights': vm.detection.FCOS_ResNet50_FPN_Weights.COCO_V1},
                "FasterRCNN_mobilenet": {'fxn': vm.detection.fasterrcnn_mobilenet_v3_large_fpn,
                                         'weights': vm.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights},
                "FasterRCNN_resnet50": {'fxn': vm.detection.faster_rcnn.fasterrcnn_resnet50_fpn_v2,
                                        'weights': vm.detection.FasterRCNN_ResNet50_FPN_V2_Weights},
                "SSD300_vgg16": {'fxn': vm.detection.ssd300_vgg16,
                                 'weights': vm.detection.SSD300_VGG16_Weights.COCO_V1},
                "RetinaNet_resnet50": {'fxn': vm.detection.retinanet_resnet50_fpn_v2,
                                 'weights': vm.detection.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1},
                "MaskRCNN_resnet50": {'fxn': vm.detection.MaskRCNN,
                                      'weights': vm.detection.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1},
                "KeyPointRCNN_resnet50": {'fxn': vm.detection.keypointrcnn_resnet50_fpn,
                                          'weights': vm.detection.KeypointRCNN_ResNet50_FPN_Weights},
                "DeepLabv3_resnet": {'fxn': vm.segmentation.deeplabv3_resnet101,
                     'weights': vm.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1},
                "FCN_resnet101": {'fxn': vm.segmentation.fcn_resnet101,
                                  'weights': vm.segmentation.FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1},
                "LRASPP": {'fxn': vm.segmentation.lraspp_mobilenet_v3_large,
                           'weights': vm.segmentation.LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1}
            }
        assert (model in self.avail_models)

        _cfg = self.avail_models[model]
        fxn = _cfg['fxn']

        self.backbone = fxn(weights=_cfg['weights']).backbone

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, imgs: torch.Tensor, imgs_sz: np.ndarray) -> torch.Tensor:
        encs = self.backbone(imgs)
        enc = self.max_pool(encs[list(encs.keys())[-1]]).flatten(1)

        enc = torch.cat([spl.max(dim=0, keepdim=True).values for spl in enc.split(imgs_sz.tolist(), dim=0)], dim=0)
        enc = enc/(enc.norm(dim=1, keepdim=True)+1e-9)
        return enc
