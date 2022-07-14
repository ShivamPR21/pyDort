from typing import List, Optional, Tuple

import numpy as np
from clort.clearn.data.dataframe import ArgoverseObjectDataFrame
from pyDort.tracking.transform_utils import bbox_3d_from_8corners

from .base import DataRepresentation


class SimpleArgoverseDetectionRepresentation(DataRepresentation):

    def __init__(self) -> None:
        pass

    @staticmethod
    def extract_state(data: ArgoverseObjectDataFrame) -> np.ndarray:
        bbox = data.bbox_global # 8 Bounding box corners in global reference frame
        feature = bbox_3d_from_8corners(bbox, data.dims)
        return feature

    def __call__(self, data: List[ArgoverseObjectDataFrame], state_aug: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if len(data) == 0:
            tmp = np.empty((0, 1), dtype=np.float32)
            return (tmp, tmp) if state_aug else (tmp, None)

        features = [self.extract_state(obj) for obj in data]
        features = np.stack(features, axis=0)

        if state_aug:
            return features, features

        return features, None
