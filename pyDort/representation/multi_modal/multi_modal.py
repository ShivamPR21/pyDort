from typing import List, Optional, Tuple, Type

import numpy as np
from clort.clearn.data.dataframe import ArgoverseObjectDataFrame

from ..base import DataRepresentation


class MultiModalRepresentation(DataRepresentation):

    def __init__(self, image_encoding_model: Type[DataRepresentation],
                 pcd_encodnig_model: Type[DataRepresentation]) -> None:
        super().__init__()
        self.im_enc = image_encoding_model
        self.pcd_enc = pcd_encodnig_model

    def __call__(self, data: List[ArgoverseObjectDataFrame], state_aug: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        im_features, state = self.im_enc(data, state_aug)
        pcd_features, _ = self.pcd_enc(data, False)

        try: #TODO: Remove excess print statements
            print(im_features.shape, pcd_features.shape)
        except:
            print("Error:", im_features, pcd_features)

        im_features /= np.linalg.norm(im_features, axis=1, keepdims=True)
        pcd_features /= np.linalg.norm(pcd_features, axis=1, keepdims=True)

        features = np.concatenate((im_features, pcd_features), axis=1)

        return features, state
