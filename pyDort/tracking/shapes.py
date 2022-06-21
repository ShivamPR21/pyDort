'''
Copyright (C) 2022  Shiavm Pandey

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import numpy as np

from .primitives import Dims3D, Pose2D, Primitives


class BoxCorners3D(Primitives):

    def __init__(self, corners: np.ndarray) -> None:
        assert(corners.shape == (8, 3))
        self.corners = corners

    def __call__(self) -> np.ndarray:
        return self.corners

    def __len__(cls) -> int:
        return 8

class BoxYaw3D(Primitives):
    def __init__(self,
                 c_x: float,
                 c_y: float,
                 c_z: float,
                 yaw: float,
                 l: float,
                 b: float,
                 h: float) -> None:
        self.pose = Pose2D(c_x, c_y, yaw)
        self.pose.z = c_z
        self.dims = Dims3D(l, b, h)

    def __call__(self) -> np.ndarray:
        out_arr = np.array([*self.pose(), self.pose.z, *self.dims()], dtype=np.float32)
        out_arr[2], out_arr[3] = out_arr[3], out_arr[2]
        return out_arr

    def __len__(cls) -> int:
        return 7
