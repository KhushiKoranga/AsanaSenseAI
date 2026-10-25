"""Load reference samples and derive robust median joint-angle targets."""
from pathlib import Path
import pandas as pd
from src.angle_analyzer import calculate_joint_angles

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train_landmarks.csv"


def load_dataset(path=DATA_PATH):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_reference(dataset, pose_name):
    if dataset.empty or "label" not in dataset.columns:
        return {}, []
    rows = dataset[dataset["label"].astype(str) == pose_name]
    required = [f"{i}_{axis}" for i in range(33) for axis in ("x", "y")]
    if rows.empty or any(col not in rows.columns for col in required):
        return {}, []
    angle_rows = []
    valid_rows = []
    for _, row in rows.iterrows():
        points = [(float(row[f"{i}_x"]), float(row[f"{i}_y"])) for i in range(33)]
        try:
            angle_rows.append(calculate_joint_angles(points))
            valid_rows.append(points)
        except (ValueError, TypeError):
            continue
    if not valid_rows:
        return {}, []
    angle_df = pd.DataFrame(angle_rows)
    median_angles = {key: float(angle_df[key].median()) for key in angle_df.columns}
    # A representative sample provides a drawable ghost pose; median angles are used for scoring.
    return median_angles, valid_rows[len(valid_rows) // 2]
