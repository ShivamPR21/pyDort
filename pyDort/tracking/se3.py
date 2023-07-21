# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""SE3 class for point cloud rotation and translation."""

import numpy as np


class SE3:
    """An SE3 class allows point cloud rotation and translation operations."""

    def __init__(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        assert rotation.shape == (3, 3)
        assert translation.shape == (3,)
        self.rotation = rotation
        self.translation = translation

        self.transform_matrix = np.eye(4)
        self.transform_matrix[:3, :3] = self.rotation
        self.transform_matrix[:3, 3] = self.translation

    def transform_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        return point_cloud @ self.rotation.T + self.translation

    def inverse_transform_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """Undo the translation and then the rotation (Inverse SE(3) transformation)."""
        return (point_cloud.copy() - self.translation) @ self.rotation

    def inverse(self) -> "SE3":
        return SE3(rotation=self.rotation.T, translation=self.rotation.T.dot(-self.translation))

    def compose(self, right_se3: "SE3") -> "SE3":
        chained_transform_matrix = self.transform_matrix @ right_se3.transform_matrix
        chained_se3 = SE3(
            rotation=chained_transform_matrix[:3, :3],
            translation=chained_transform_matrix[:3, 3],
        )
        return chained_se3

    def right_multiply_with_se3(self, right_se3: "SE3") -> "SE3":
        return self.compose(right_se3)
