"""Train/load a Random Forest classifier for landmark-based pose labels."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.feature_extraction import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "pose_classifier.joblib"
ENCODER_PATH = ROOT / "models" / "label_encoder.joblib"


def train_model(csv_path, model_path=MODEL_PATH, encoder_path=ENCODER_PATH):
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLUMNS + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing[:5]}")
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"])
    if df["label"].nunique() < 2:
        raise ValueError("Training requires at least two different pose classes.")
    X = df[FEATURE_COLUMNS].astype(np.float32)
    y = df["label"].astype(str)
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify)
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(FEATURE_COLUMNS, Path(encoder_path).with_name("feature_columns.joblib"))
    print(f"Test accuracy: {accuracy:.2%}")
    print(classification_report(y_test, model.predict(X_test), zero_division=0))
    return model, accuracy

class PoseClassifier:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = (
            joblib.load(self.model_path)
            if self.model_path.exists()
            else None
        )

    @property
    def ready(self):
        return self.model is not None

    def predict(self, features):
        if self.model is None:
            return "Model not trained", 0.0

        values = np.asarray(
            features, dtype=np.float32
        ).reshape(1, -1)

        expected_features = getattr(
            self.model,
            "n_features_in_",
            values.shape[1],
        )

        if values.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} features, "
                f"received {values.shape[1]}."
            )

        if hasattr(self.model, "feature_names_in_"):
            columns = list(self.model.feature_names_in_)
            x = pd.DataFrame(values, columns=columns)
        else:
            x = values

        prediction = str(self.model.predict(x)[0])
        probabilities = self.model.predict_proba(x)[0]
        confidence = float(np.max(probabilities) * 100)

        return prediction, confidence

