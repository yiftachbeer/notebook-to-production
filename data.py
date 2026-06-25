from pathlib import Path
from typing import TypedDict

import numpy as np
from PIL import Image


class Frame(TypedDict):
    id: str
    image: Image.Image
    P2: np.ndarray


def read_P2(calib_path: str | Path) -> np.ndarray:
    # P2 is the projection matrix of the left color camera in the KITTI calib.
    for line in open(calib_path):
        if line.startswith("P2:"):
            return np.array(line.split()[1:], dtype=np.float32).reshape(3, 4)


class StereoDataset:
    """A folder of driving frames: each frame is an image + its calibration."""

    def __init__(self, root: str | Path):
        root = Path(root)
        self.paths = {p.stem: p for p in sorted((root / "left").glob("*.png"))}
        self.calib = root / "calib"
        self.ids = list(self.paths)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, frame_id: str) -> Frame:
        path = self.paths[frame_id]
        return {
            "id": frame_id,
            "image": Image.open(path).convert("RGB"),
            "P2": read_P2(self.calib / f"{frame_id}.txt"),
        }
