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

from .primitives import Dims3D, Pose2D, Pose3D, Primitives


# Vehicle state
class VehicleStateCV(Primitives):
    def __init__(self, x: float, y: float, z: float,
                 yaw: float, l:float, b: float, h: float,
                 v: float, v_z: float) -> None:
        self.pose = Pose2D(x, y, yaw)
        self.dims = Dims3D(l, b, h)
        self.pose.z = z
        self.v, self.v_z = v, v_z

    def __call__(self) -> np.ndarray:
        out_arr = np.array([*self.pose(), self.pose.z, *self.dims(), self.v, self.v_z], dtype=np.float32)
        out_arr[2], out_arr[3] = out_arr[3], out_arr[2]
        return out_arr

    def __len__(cls):
        return 8


# Vehicle state CTRV
class VehicleStateCTRV(Primitives):
    def __init__(self, x: float, y: float, z: float,
                 yaw: float, l:float, b: float, h: float,
                 v: float, v_z: float, psi_dot: float) -> None:
        self.pose = Pose2D(x, y, yaw)
        self.dims = Dims3D(l, b, h)
        self.pose.z = z
        self.v, self.v_z, self.psi_dot = v, v_z, psi_dot

    def __call__(self) -> np.ndarray:
        out_arr = np.array([*self.pose(), self.pose.z, *self.dims(), self.v, self.v_z, self.psi_dot], dtype=np.float32)
        out_arr[2], out_arr[3] = out_arr[3], out_arr[2]
        return out_arr

    def __len__(cls):
        return 9


# Pedestrian state
class PedestrianStateCV(Primitives):
    def __init__(self, x: float, y: float, z: float,
                 yaw: float, l:float, b: float, h: float,
                 v: float, v_z: float) -> None:
        self.pose = Pose2D(x, y, yaw)
        self.dims = Dims3D(l, b, h)
        self.pose.z = z
        self.v, self.v_z = v, v_z

    def __call__(self) -> np.ndarray:
        out_arr = np.array([*self.pose(), self.pose.z, *self.dims(), self.v, self.v_z], dtype=np.float32)
        out_arr[2], out_arr[3] = out_arr[3], out_arr[2]
        return out_arr

    def __len__(cls):
        return 8


# Random object
class RandomObjectState(Primitives):
    def __init__(self, x: float, y: float, z: float,
                 roll: float, pitch: float, yaw: float,
                 l:float, b: float, h: float) -> None:
        self.pose = Pose3D(x, y, z, roll, pitch, yaw)
        self.dims = Dims3D(l, b, h)

    def __call__(self) -> np.ndarray:
        return np.array([*self.pose(), *self.dims()], dtype=np.float32)

    def __len__(cls):
        return 6
