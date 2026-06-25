from typing import Protocol

import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from data import Frame
from device import DEVICE, DEVICE_T, DTYPE


class DepthEstimator(Protocol):
    def predict(self, frame: Frame) -> np.ndarray: ...


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


class UniDepth:
    """A different method with a different call: it consumes the camera intrinsics.
    Same predict(frame) signature, so the pipeline doesn't care which one ran."""

    def __init__(self, version: str = "v2", backbone: str = "vitl14"):
        self.model = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth",
                                    version=version, backbone=backbone,
                                    pretrained=True, trust_repo=True).to(DEVICE_T).eval()

    def predict(self, frame: Frame) -> np.ndarray:
        rgb = torch.from_numpy(np.array(frame["image"])).permute(2, 0, 1).to(DEVICE_T)
        K = torch.from_numpy(frame["P2"][:, :3]).to(DEVICE_T)
        return self.model.infer(rgb, K)["depth"].squeeze().cpu().numpy()
