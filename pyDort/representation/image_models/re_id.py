import os
from pathlib import Path
from typing import Any, Dict

import gdown
import numpy as np
import torch
import torch.nn as nn
from torchreid.models import build_model


class ReIdRepresentation(nn.Module):

    def __init__(self, model:str = "osnet_x1_0") -> None:
        super().__init__()
        self.cache_dir = f'{Path(__file__).parent.resolve()}/chkpts'

        self.avail_models: Dict[str, Any] = \
            {
                "osnet_x1_0": {'url': 'https://drive.google.com/file/d/1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA/view?usp=sharing',
                                'cached_file': 'osnet_x1_0_Marker1501.pth',
                                'out_dim': 512},
                "osnet_ain_x1_0_msmt":{'url': 'https://drive.google.com/file/d/1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal/view?usp=sharing',
                                   'cached_file': 'osnet_ain_x1_0_MSMT.pth',
                                   'out_dim': 512}
            }
        assert (model in self.avail_models)


        _cfg = self.avail_models[model]

        out_path = os.path.join(self.cache_dir, _cfg['cached_file'])
        if not os.path.exists(out_path):
            gdown.download(url=_cfg['url'],
                        output=out_path,
                        quiet=False, fuzzy=True)

        self.model : nn.Module | None = None
        if model == "osnet_x1_0":
            self.model = build_model("osnet_x1_0", 100, loss = "")
            self.model.classifier = None
        elif model == "osnet_ain_x1_0_msmt":
            self.model = build_model("osnet_ain_x1_0", 100, loss="")
            self.model.classifier = None
        else:
            raise NotImplementedError

        ckpt = torch.load(out_path)
        print(f'{self.model.load_state_dict(ckpt, strict=False) = }')

    def forward(self, imgs: torch.Tensor, imgs_sz: np.ndarray) -> torch.Tensor:
        enc = self.model(imgs).flatten(1)

        enc = torch.cat([spl.max(dim=0, keepdim=True).values for spl in enc.split(imgs_sz.tolist(), dim=0)], dim=0)
        enc = enc/(enc.norm(dim=1, keepdim=True)+1e-9)
        return enc
