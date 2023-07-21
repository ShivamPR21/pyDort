# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

"""Module for `SE2`."""

import numpy as np


class SE2:
    def __init__(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        assert(rotation.shape == (2, 2))
        assert(translation.shape == (2,))
        self.rotation = rotation
        self.translation = translation
        self.transform_matrix = np.eye(3)
        self.transform_matrix[:2, :2] = self.rotation
        self.transform_matrix[:2, 2] = self.translation

    def transform_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        assert(point_cloud.shape[1] == 2)
        num_points = point_cloud.shape[0]
        homogeneous_pts = np.hstack([point_cloud, np.ones((num_points, 1))])
        transformed_point_cloud = homogeneous_pts.dot(self.transform_matrix.T)
        return transformed_point_cloud[:, :2]

    def inverse(self) -> "SE2":
        return SE2(rotation=self.rotation.T, translation=self.rotation.T.dot(-self.translation))

    def inverse_transform_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        return self.inverse().transform_point_cloud(point_cloud)

    def right_multiply_with_se2(self, right_se2: "SE2") -> "SE2":
        chained_transform_matrix = self.transform_matrix.dot(right_se2.transform_matrix)
        chained_se2 = SE2(
            rotation=chained_transform_matrix[:2, :2],
            translation=chained_transform_matrix[:2, 2],
        )
        return chained_se2
