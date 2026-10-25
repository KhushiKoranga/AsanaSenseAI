"""Collect labelled landmark samples from the webcam into data/train_landmarks.csv."""
import argparse
import csv
from pathlib import Path
import cv2

from src.pose_detection import PoseDetector
from src.feature_extraction import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent
POSES = ["Bound_Angle_Pose", "Cobra_Pose", "Downdog_Pose", "Goddess_Pose",
         "Plank_Pose", "Tree_Pose", "Warrior2_Pose"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", required=True, choices=POSES)
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--output", default=str(ROOT / "data" / "train_landmarks.csv"))
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existed = output.exists() and output.stat().st_size > 0
    camera = cv2.VideoCapture(0)
    detector = PoseDetector()
    count = 0
    frame_tick = 0
    try:
        with output.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not existed:
                writer.writerow(FEATURE_COLUMNS + ["label"])
            while count < args.samples:
                ok, frame = camera.read()
                if not ok:
                    print("Could not read webcam frame.")
                    break
                frame = cv2.flip(frame, 1)
                result = detector.detect(frame)
                if result and result.pose_landmarks:
                    detector.draw(frame, result)
                    frame_tick += 1
                    if frame_tick % 4 == 0:
                        values = []
                        for lm in result.pose_landmarks.landmark:
                            values.extend([lm.x, lm.y])
                        writer.writerow(values + [args.pose])
                        count += 1
                cv2.putText(frame, f"Pose: {args.pose} | Saved: {count}/{args.samples}",
                            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)
                cv2.putText(frame, "Hold the pose; press Q to stop", (15, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.imshow("AsanaSenseAI - Dataset Collection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        detector.close()
    print(f"Saved {count} samples to {output}")


if __name__ == "__main__":
    main()
