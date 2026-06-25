import numpy as np

N_POINTS = 60000


def backproject(
    depth_m: np.ndarray,
    seg_color: np.ndarray,
    P2: np.ndarray,
    n_points: int = N_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift each pixel to 3D with the real intrinsics, colored by its class."""
    H, W = depth_m.shape
    fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]
    # depth and segmentation are both native-res, so they line up pixel-for-pixel
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth_m
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    colors = seg_color.reshape(-1, 3)

    idx = np.random.choice(len(points), min(n_points, len(points)), replace=False)
    return points[idx], colors[idx]
