import numpy as np

from ..helpers import wrap_angle


def get_distance_iou_3d(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Note this is not traditional 2d or 3d iou, but rather we align two cuboids
    along their x-axes, and compare 3d volume differences.
    """
    w1 = x1[1]
    l1 = x1[0]
    h1 = x1[2]

    w2 = x2[1]
    l2 = x2[0]
    h2 = x2[2]

    x_overlap = max(0, min(l1 / 2, l2 / 2) - max(-l1 / 2, -l2 / 2))
    y_overlap = max(0, min(w1 / 2, w2 / 2) - max(-w1 / 2, -w2 / 2))
    overlapArea = x_overlap * y_overlap
    inter = overlapArea * min(h1, h2)
    union = w1 * l1 * h1 + w2 * l2 * h2 - inter
    score = 1 - inter / union

    return float(score)


def get_distance(x1: np.ndarray, x2: np.ndarray, name: str) -> float:
    if name == "centroid":
        dist = float(np.linalg.norm(x1[0:3] - x2[0:3]))
        # return dist if dist < 50 else float(np.nan)
        return dist
    elif name == "iou":
        return get_distance_iou_3d(x1[4:7], x2[4:7])
    elif name == "orientation":
        theta = np.array([x1[3]] - x2[3])
        dist = wrap_angle(theta).item()

        # Convert to degrees.
        return float(np.rad2deg(dist))
    else:
        raise NotImplementedError("Not implemented..")
