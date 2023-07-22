#!/usr/bin/env python3

import copy

import numpy as np
from scipy.spatial.transform import Rotation

from .se2 import SE2
from .se3 import SE3


def yaw_to_quaternion3d(yaw: float) -> np.ndarray:
    # """Convert a rotation angle in the xy plane (i.e. about the z axis) to a quaternion.

    # Args:
    #     yaw: angle to rotate about the z-axis, representing an Euler angle, in radians

    # Returns:
    #     array w/ quaternion coefficients (qw,qx,qy,qz) in scalar-first order, per Argoverse convention.
    # """
    qx, qy, qz, qw = Rotation.from_euler(seq="z", angles=yaw, degrees=False).as_quat(canonical=True)
    return np.array([qw, qx, qy, qz])


def rotmat2quat(R: np.ndarray) -> np.ndarray:
    """Convert a rotation-matrix to a quaternion in Argo's scalar-first notation (w, x, y, z)."""
    quat_xyzw = Rotation.from_matrix(R).as_quat(canonical=True)
    quat_wxyz = quat_scipy2argo(quat_xyzw)
    return quat_wxyz


def quat2rotmat(q: np.ndarray) -> np.ndarray:
    # """Normalizes a quaternion to unit-length, then converts it into a rotation matrix.

    # Note that libraries such as Scipy expect a quaternion in scalar-last [x, y, z, w] format,
    # whereas at Argo we work with scalar-first [w, x, y, z] format, so we convert between the
    # two formats here. We use the [w, x, y, z] order because this corresponds to the
    # multidimensional complex number `w + ix + jy + kz`.

    # Args:
    #     q: Array of shape (4,) representing (w, x, y, z) coordinates

    # Returns:
    #     R: Array of shape (3, 3) representing a rotation matrix.
    # """
    norm = np.linalg.norm(q)
    if not np.isclose(norm, 1.0, atol=1e-12):
        # logger.info("Forced to re-normalize quaternion, since its norm was not equal to 1.")
        if np.isclose(norm, 0.0):
            raise ZeroDivisionError("Normalize quaternioning with norm=0 would lead to division by zero.")
        q /= norm

    quat_xyzw = quat_argo2scipy(q)
    return Rotation.from_quat(quat_xyzw).as_matrix()


def quat_argo2scipy(q: np.ndarray) -> np.ndarray:
    # """Re-order Argoverse's scalar-first [w,x,y,z] quaternion order to Scipy's scalar-last [x,y,z,w]"""
    w, x, y, z = q
    q_scipy = np.array([x, y, z, w])
    return q_scipy


def quat_scipy2argo(q: np.ndarray) -> np.ndarray:
    # """Re-order Scipy's scalar-last [x,y,z,w] quaternion order to Argoverse's scalar-first [w,x,y,z]."""
    x, y, z, w = q
    q_argo = np.array([w, x, y, z])
    return q_argo


def quat_argo2scipy_vectorized(q: np.ndarray) -> np.ndarray:
    # """Re-order Argoverse's scalar-first [w,x,y,z] quaternion order to Scipy's scalar-last [x,y,z,w]"""
    return q[..., [1, 2, 3, 0]]


def quat_scipy2argo_vectorized(q: np.ndarray) -> np.ndarray:
    # """Re-order Scipy's scalar-last [x,y,z,w] quaternion order to Argoverse's scalar-first [w,x,y,z]."""
    return q[..., [3, 0, 1, 2]]

def rotmat2d(theta: float) -> np.ndarray:
    # """
    #     Return rotation matrix corresponding to rotation theta.

    #     Args:
    #     -   theta: rotation amount in radians.

    #     Returns:
    #     -   R: 2 x 2 np.ndarray rotation matrix corresponding to rotation theta.
    # """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R = np.array(
        [
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
    return R

def get_B_SE2_A(B_SE3_A: SE3):
    # """
    #     Can take city_SE3_egovehicle -> city_SE2_egovehicle
    #     Can take egovehicle_SE3_object -> egovehicle_SE2_object

    #     Doesn't matter if we stretch square by h,w,l since
    #     triangles will be similar regardless

    #     Args:
    #     -   B_SE3_A

    #     Returns:
    #     -   B_SE2_A
    #     -   B_yaw_A
    # """
    x_corners = np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners_A_frame = np.vstack((x_corners, y_corners, z_corners)).T

    corners_B_frame = B_SE3_A.transform_point_cloud(corners_A_frame)

    p1 = corners_B_frame[1]
    p5 = corners_B_frame[5]
    dy = p1[1] - p5[1]
    dx = p1[0] - p5[0]
    # the orientation angle of the car
    B_yaw_A = np.arctan2(dy, dx)

    t = B_SE3_A.transform_matrix[:2,3] # get x,y only
    B_SE2_A = SE2(
        rotation=rotmat2d(B_yaw_A),
        translation=t
    )
    return B_SE2_A, B_yaw_A

def se2_to_yaw(B_SE2_A):
    # """
    # Computes the pose vector v from a homogeneous transform A.
    # Args:
    # -   B_SE2_A
    # Returns:
    # -   v
    # """
    R = B_SE2_A.rotation
    theta = np.arctan2(R[1,0], R[0,0])
    return theta

def test_yaw_to_quaternion3d():
    for i, yaw in enumerate(np.linspace(0, 3*np.pi, 50)):
        print(f'On iter {i}')
        dcm = rotMatZ_3D(yaw)
        qx,qy,qz,qw = Rotation.from_matrix(dcm).as_quat(canonical=True)

        qw_, qx_, qy_, qz_ = yaw_to_quaternion3d(yaw)
        print(qx_, qy_, qz_, qw_, ' vs ', qx,qy,qz,qw)
        assert np.allclose(qx, qx_, atol=1e-3)
        assert np.allclose(qy, qy_, atol=1e-3)
        assert np.allclose(qz, qz_, atol=1e-3)
        assert np.allclose(qw, qw_, atol=1e-3)

def roty(t: float):
    # """
    # Compute rotation matrix about the y-axis.
    # """
    c = np.cos(t)
    s = np.sin(t)
    R = np.array(
        [
            [c,  0,  s],
            [0,  1,  0],
            [-s, 0,  c]
        ])
    return R

def rotMatZ_3D(yaw):
    # """
    #     Args:
    #     -   tz

    #     Returns:
    #     -   rot_z
    # """
    # c = np.cos(yaw)
    # s = np.sin(yaw)
    # rot_z = np.array(
    #     [
    #         [   c,-s, 0],
    #         [   s, c, 0],
    #         [   0, 0, 1 ]
    #     ])

    rot_z = Rotation.from_euler('z', yaw).as_matrix()
    return rot_z

def convert_3dbox_to_8corner(bbox3d_input: np.ndarray) -> np.ndarray:
    # '''
    #     Args:
    #     -   bbox3d_input: Numpy array of shape (7,) representing
    #             tx,ty,tz,yaw,l,w,h. (tx,ty,tz,yaw) tell us how to
    #             transform points to get from the object frame to
    #             the egovehicle frame.

    #     Returns:
    #     -   corners_3d: (8,3) array in egovehicle frame
    # '''
    # compute rotational matrix around yaw axis
    bbox3d = copy.copy(bbox3d_input)
    yaw = bbox3d[3]
    t = bbox3d[:3]

    # 3d bounding box dimensions
    l = bbox3d[4]
    w = bbox3d[5]
    h = bbox3d[6]

    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    z_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]

    # rotate and translate 3d bounding box
    corners_3d_obj_fr = np.vstack([x_corners,y_corners,z_corners]).T
    egovehicle_SE3_object = SE3(rotation=rotMatZ_3D(yaw), translation=t)
    corners_3d_ego_fr = egovehicle_SE3_object.transform_point_cloud(corners_3d_obj_fr)
    return corners_3d_ego_fr

def yaw_from_bbox_corners(det_corners: np.ndarray) -> float:
    # """
    # Use basic trigonometry on cuboid to get orientation angle.
    #     Args:
    #     -   det_corners: corners of bounding box
    #     Returns:
    #     -   yaw
    # """
    p1 = det_corners[1]
    p5 = det_corners[5]
    dy = p1[1] - p5[1]
    dx = p1[0] - p5[0]
    # the orientation angle of the car
    yaw = np.arctan2(dy, dx)
    return yaw

def bbox_dims(det_corners: np.ndarray) -> np.ndarray:
    height = np.linalg.norm(det_corners[[2, 3, 6, 7], :].mean(axis=0) - det_corners[[0, 1, 4, 5], :].mean(axis=0))
    length = np.linalg.norm(det_corners[:4, :].mean(axis=0) - det_corners[4:, :].mean(axis=0))
    width = np.linalg.norm(det_corners[[1, 2, 5, 6], :].mean(axis=0) - det_corners[[0, 3, 4, 7], :].mean(axis=0))
    return np.array([length, width, height], dtype=np.float32)

def bbox_3d_from_8corners(det_corners:np.ndarray) -> np.ndarray:
    ego_xyz = np.mean(det_corners, axis=0)

    yaw = yaw_from_bbox_corners(det_corners)
    bbox = np.array([ego_xyz[0], ego_xyz[1], ego_xyz[2], yaw, *(bbox_dims(det_corners).tolist())], dtype=np.float32)
    return bbox

def batch_bbox_3d_from_8corners(det_corners:np.ndarray) -> np.ndarray:
    assert(det_corners.ndim == 3)
    batch_size = det_corners.shape[0]

    bboxs = np.zeros((batch_size, 7), dtype=np.float32)
    for i in range(batch_size):
        bboxs[i, :] = bbox_3d_from_8corners(det_corners[i, :, :])

    return bboxs

def angle_constraint(theta:float) -> float:
    if theta >= np.pi:
        return (theta - np.pi * 2)    # make the theta still in the range
    if theta < -np.pi:
        return (theta + np.pi * 2)

    return theta

if __name__ == '__main__':
    test_yaw_to_quaternion3d()
