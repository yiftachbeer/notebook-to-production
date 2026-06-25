import csv

from data import StereoDataset
from models import HfDepthModel, HfSegmentationModel
from metrics import frame_metrics

DATA_DIR = "stereo_data"
METRICS_CSV = "metrics.csv"


def main():
    frames = StereoDataset(DATA_DIR)
    depth_model = HfDepthModel()
    segmentation_model = HfSegmentationModel()

    rows = []
    for frame_id in frames.ids:
        frame = frames[frame_id]
        depth_m = depth_model.predict(frame)
        seg_color = segmentation_model.predict(frame)
        m = frame_metrics(frame_id, depth_m, seg_color)
        rows.append(m)
        print(f"{m['id']}  valid={m['valid_frac']:.2f}  "
              f"p50={m['depth_p50']:5.1f}  max={m['depth_max']:5.1f}  road={m['road_frac']:.2f}")

    cols = list(rows[0])
    with open(METRICS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    summary = {k: sum(r[k] for r in rows) / len(rows) for k in cols if k != "id"}
    print("mean:", {k: round(v, 3) for k, v in summary.items()})
    print(f"wrote {METRICS_CSV} ({len(rows)} frames)")


if __name__ == "__main__":
    main()
