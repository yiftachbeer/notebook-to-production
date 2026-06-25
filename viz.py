import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image


def show_image(image: Image.Image, title: str = "input"):
    plt.figure(figsize=(12, 4))
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()


def show_depth(depth_m: np.ndarray):
    plt.figure(figsize=(12, 4))
    plt.imshow(depth_m, cmap="magma")
    plt.axis("off")
    plt.title("depth (m)")
    plt.colorbar(fraction=0.025)
    plt.show()


def show_seg(seg_color: np.ndarray):
    plt.figure(figsize=(12, 4))
    plt.imshow(seg_color)
    plt.axis("off")
    plt.title("segmentation")
    plt.show()


def show_cloud(points: np.ndarray, colors: np.ndarray):
    rgb = ["rgb(%d,%d,%d)" % (r, g, b) for r, g, b in colors]
    fig = go.Figure(go.Scatter3d(
        x=points[:, 0], y=-points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(size=1.5, color=rgb, opacity=1.0),
    ))
    fig.update_layout(
        template="plotly_dark",
        width=1000, height=650,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=0.0, y=0.35, z=-1.7), up=dict(x=0, y=1, z=0)),
        ),
    )
    fig.show()
