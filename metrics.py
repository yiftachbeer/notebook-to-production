import numpy as np

from models import CITYSCAPES_COLORS


def frame_metrics(frame_id: str, depth_m: np.ndarray, seg_color: np.ndarray) -> dict[str, str | float]:
    """Proxy metrics that flag a broken frame without anyone looking at it."""
    finite = np.isfinite(depth_m)
    in_range = finite & (depth_m > 0) & (depth_m < 80)  # 0-80 m is a sane road scene

    seg = seg_color.reshape(-1, 3)
    road = np.all(seg == CITYSCAPES_COLORS["road"], axis=1)

    return {
        "id": frame_id,
        "valid_frac": float(in_range.mean()),       # sky / road blow-ups drop this
        "depth_p50": float(np.median(depth_m[in_range])),  # wrong scale shows here
        "depth_max": float(depth_m[finite].max()),
        "road_frac": float(road.mean()),            # segmentation + geometry sanity
    }
