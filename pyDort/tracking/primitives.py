'''
Copyright (C) 2021  Shiavm Pandey

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


from typing import Tuple

import numpy as np


# Single dimension
class Pose1D():
    x : float

    def __init__(self, x: float) -> None:
        self.x = x

    def __call__(self) -> np.ndarray:
        return np.array([self.x], dtype=np.float32)

class Dims1D():
    l : float

    def __init__(self, l: float) -> None:
        self.l = l

    def __call__(self) -> np.ndarray:
        return np.array([self.l], dtype=np.float32)

class PoseDot1D():
    x_dot : float

    def __init__(self, x_dot: float) -> None:
        self.x_dot = x_dot

    def __call__(self) -> np.ndarray:
        return np.array([self.x_dot], dtype=np.float32)

class DimsDot1D():
    l_dot : float

    def __init__(self, l_dot: float) -> None:
        self.l_dot = l_dot

    def __call__(self) -> np.ndarray:
        return np.array([self.l_dot], dtype=np.float32)

class Object1D():
    pose : Pose1D
    dims : Dims1D

    def __init__(self, pose: Pose1D, dims: Dims1D) -> None:
        self.pose, self.dims = pose, dims

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.pose(), self.dims()

class ObjectDot1D():
    pose_dot : PoseDot1D
    dims_dot : DimsDot1D

    def __init__(self, pose_dot: PoseDot1D, dims_dot: DimsDot1D) -> None:
        self.pose_dot, self.dims_dot = pose_dot, dims_dot

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.pose_dot(), self.dims_dot()


# 2D types
class Pose2D(Pose1D):
    y : float
    theta : float

    def __init__(self, x: float, y: float, yaw: float) -> None:
        super().__init__(x)

        self.y = y

        # Rotation
        self.yaw = yaw

    def __call__(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw], dtype=np.float32)

class Dims2D(Dims1D):
    b : float

    def __init__(self, l: float, b: float) -> None:
        super().__init__(l)

        self.b = b

    def __call__(self) -> np.ndarray:
        return np.array([self.l, self.b], dtype=np.float32)

class PoseDot2D(PoseDot1D):
    y_dot : float

    def __init__(self, x_dot: float, y_dot: float, yaw_dot: float) -> None:
        super().__init__(x_dot)

        self.y_dot = y_dot

        # Rotation
        self.yaw_dot = yaw_dot

    def __call__(self) -> np.ndarray:
        return np.array([self.x_dot, self.y_dot, self.yaw_dot], dtype=np.float32)

class DimsDot2D(DimsDot1D):
    b_dot : float

    def __init__(self, l_dot: float, b_dot: float) -> None:
        super().__init__(l_dot)

        self.b_dot = b_dot

    def __call__(self) -> np.ndarray:
        return np.array([self.l_dot, self.b_dot], dtype=np.float32)

class Object2D():
    pose : Pose2D
    dims : Dims2D

    def __init__(self, pose: Pose2D, dims: Dims2D) -> None:
        self.pose, self.dims = pose, dims

    def __call__(self) -> np.ndarray:
        return np.concatenate((self.pose(), self.dims()))


class ObjectDot2D():
    pose_dot : PoseDot2D
    dims_dot : DimsDot2D

    def __init__(self, pose_dot: PoseDot2D, dims_dot: DimsDot2D) -> None:
        self.pose_dot, self.dims_dot = pose_dot, dims_dot

    def __call__(self) -> np.ndarray:
        return np.concatenate((self.pose_dot(), self.dims_dot()))

# 3D types
class Pose3D(Pose2D):
    z : float

    def __init__(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> None:
        super().__init__(x, y)

        self.z = z

        # Rotation
        self.roll, self.pitch, self.yaw = roll, pitch, yaw

    def __call__(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z,
                         self.roll, self.pitch, self.yaw], dtype=np.float32)

class Dims3D(Dims2D):
    h : float

    def __init__(self, l: float, b: float, h: float) -> None:
        super().__init__(l, b)

        self.h = h

    def __call__(self) -> np.ndarray:
        return np.array([self.l, self.b, self.h], dtype=np.float32)


class PoseDot3D(PoseDot2D):
    z_dot : float

    def __init__(self, x_dot: float, y_dot: float, z_dot: float, roll_dot: float, pitch_dot: float, yaw_dot: float) -> None:
        super().__init__(x_dot, y_dot)

        self.z_dot = z_dot

        # Rotation
        self.roll_dot, self.pitch_dot, self.yaw_dot = roll_dot, pitch_dot, yaw_dot

    def __call__(self) -> np.ndarray:
        return np.array([self.x_dot, self.y_dot, self.z_dot,
                         self.roll_dot, self.pitch_dot, self.yaw_dot], dtype=np.float32)


class DimsDot3D(DimsDot2D):
    h_dot : float

    def __init__(self, l_dot: float, b_dot: float, h_dot: float) -> None:
        super().__init__(l_dot, b_dot)

        self.h_dot = h_dot

    def __call__(self) -> np.ndarray:
        return np.array([self.l_dot, self.b_dot, self.h_dot], dtype=np.float32)


class Object3D():
    pose : Pose3D
    dims : Dims3D

    def __init__(self, pose: Pose3D, dims: Dims3D) -> None:
        self.pose, self.dims = pose, dims

    def __call__(self) -> np.ndarray:
        return np.array([*self.pose(), *self.dims()], dtype=np.float32)


class ObjectDot3D():
    pose_dot : PoseDot3D
    dims_dot : DimsDot3D

    def __init__(self, pose_dot: PoseDot3D, dims_dot: DimsDot3D) -> None:
        self.pose_dot, self.dims_dot = pose_dot, dims_dot

    def __call__(self) -> np.ndarray:
        return np.array([*self.pose_dot(), *self.dims_dot()], dtype=np.float32)


# Vehicle state
class VehicleState():
    pose : Pose2D
    dims : Dims3D

    def __init__(self, x: float, y: float, z: float, yaw: float, l:float, b: float, h: float) -> None:
        self.pose = Pose2D(x, y, yaw)
        self.dims = Dims3D(l, b, h)
        self.pose.z = z

    def __call__(self) -> np.ndarray:
        out_arr = np.array([*self.pose(), self.pose.z, *self.dims()], dtype=np.float32)
        out_arr[2], out_arr[3] = out_arr[3], out_arr[2]
        return out_arr

# Pedestrian state
class PedestrianState():
    pose : Pose2D
    dims : Dims3D

    def __init__(self, x: float, y: float, z: float, yaw: float, l:float, b: float, h: float) -> None:
        self.pose = Pose2D(x, y, yaw)
        self.dims = Dims3D(l, b, h)
        self.pose.z = z

    def __call__(self) -> np.ndarray:
        out_arr = np.array([*self.pose(), self.pose.z, *self.dims()], dtype=np.float32)
        out_arr[2], out_arr[3] = out_arr[3], out_arr[2]
        return out_arr

# Random object
class RandomObjectState():
    pose : Pose3D
    dims : Dims3D

    def __init__(self, x: float, y: float, z: float,
                 roll: float, pitch: float, yaw: float,
                 l:float, b: float, h: float) -> None:
        self.pose = Pose3D(x, y, z, roll, pitch, yaw)
        self.dims = Dims3D(l, b, h)

    def __call__(self) -> np.ndarray:
        return np.array([*self.pose(), *self.dims()], dtype=np.float32)
