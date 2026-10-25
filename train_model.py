from src.pose_classifier import train_model
from src.reference_pose import DATA_PATH

if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise SystemExit(
            "Dataset not found. Collect samples first using collect_data.py."
        )

    train_model(DATA_PATH)