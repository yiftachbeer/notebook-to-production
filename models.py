import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from data import Frame

DEVICE = 0 if torch.cuda.is_available() else -1
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

CITYSCAPES_COLORS = {
    "road": (128, 64, 128), "sidewalk": (244, 35, 232), "building": (70, 70, 70),
    "wall": (102, 102, 156), "fence": (190, 153, 153), "pole": (153, 153, 153),
    "traffic light": (250, 170, 30), "traffic sign": (220, 220, 0),
    "vegetation": (107, 142, 35), "terrain": (152, 251, 152), "sky": (70, 130, 180),
    "person": (220, 20, 60), "rider": (255, 0, 0), "car": (0, 0, 142),
    "truck": (0, 0, 70), "bus": (0, 60, 100), "train": (0, 80, 100),
    "motorcycle": (0, 0, 230), "bicycle": (119, 11, 32),
}


class HfDepthModel:
    """Monocular metric-depth model. predict(frame) -> (H, W) depth in meters."""

    def __init__(self, model_id: str, revision: str | None = None):
        self.pipe = pipeline("depth-estimation", model=model_id, revision=revision,
                             device=DEVICE, torch_dtype=DTYPE)

    def predict(self, frame: Frame) -> np.ndarray:
        image = frame["image"]
        W, H = image.size
        depth = self.pipe(image)["predicted_depth"].squeeze().float().cpu().numpy()
        if depth.shape != (H, W):
            depth = np.array(Image.fromarray(depth).resize((W, H), Image.BILINEAR))
        return depth


class HfSegmentationModel:
    """Semantic segmentation. predict(frame) -> (H, W, 3) Cityscapes color map."""

    def __init__(self, model_id: str, revision: str | None = None):
        self.pipe = pipeline("image-segmentation", model=model_id, revision=revision,
                             device=DEVICE, torch_dtype=DTYPE)

    def predict(self, frame: Frame) -> np.ndarray:
        image = frame["image"]
        W, H = image.size
        seg_color = np.zeros((H, W, 3), dtype=np.uint8)
        for s in self.pipe(image):
            mask = np.array(s["mask"]) > 0
            seg_color[mask] = CITYSCAPES_COLORS.get(s["label"], (0, 0, 0))
        return seg_color
