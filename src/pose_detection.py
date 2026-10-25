
"""Real-time yoga pose detection using MediaPipe Tasks API."""
from pathlib import Path
import cv2
import mediapipe as mp


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"


class PoseDetector:
    """Detect, validate, and draw pose landmarks."""

    def __init__(self, model_path=DEFAULT_MODEL):
        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Pose model not found: {self.model_path}"
            )

        vision = mp.tasks.vision

        options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self.model_path)
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(
            options
        )
        self.timestamp_ms = 0
        self.visibility_threshold = 0.65

        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12),
            (11, 13), (13, 15), (15, 17), (15, 19),
            (15, 21), (17, 19),
            (12, 14), (14, 16), (16, 18), (16, 20),
            (16, 22), (18, 20),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (27, 29), (29, 31),
            (27, 31), (24, 26), (26, 28), (28, 30),
            (30, 32), (28, 32),
        ]

    def detect(self, frame):
        """Detect landmarks from an OpenCV BGR frame."""
        if frame is None:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        self.timestamp_ms += 33

        return self.landmarker.detect_for_video(
            mp_image,
            self.timestamp_ms,
        )

    @staticmethod
    def get_landmarks(results):
        """Return the first detected pose."""
        if results is None or not results.pose_landmarks:
            return None

        return results.pose_landmarks[0]

    def is_visible(self, landmark):
        """Check landmark visibility and image boundaries."""
        visibility = getattr(landmark, "visibility", 0.0)
        presence = getattr(landmark, "presence", 1.0)

        return (
            visibility >= self.visibility_threshold
            and presence >= 0.5
            and 0.0 <= landmark.x <= 1.0
            and 0.0 <= landmark.y <= 1.0
        )

    def draw_landmarks(self, frame, results):
        """Draw only sufficiently visible landmarks and connections."""
        landmarks = self.get_landmarks(results)

        if landmarks is None:
            return frame

        height, width = frame.shape[:2]

        for start, end in self.connections:
            if not (
                self.is_visible(landmarks[start])
                and self.is_visible(landmarks[end])
            ):
                continue

            p1 = (
                int(landmarks[start].x * width),
                int(landmarks[start].y * height),
            )
            p2 = (
                int(landmarks[end].x * width),
                int(landmarks[end].y * height),
            )

            cv2.line(
                frame, p1, p2, (0, 255, 0), 2, cv2.LINE_AA
            )

        for landmark in landmarks:
            if not self.is_visible(landmark):
                continue

            point = (
                int(landmark.x * width),
                int(landmark.y * height),
            )

            cv2.circle(
                frame, point, 4, (0, 0, 255), -1, cv2.LINE_AA
            )

        return frame

    def close(self):
        """Release the MediaPipe landmarker."""
        landmarker = getattr(self, "landmarker", None)

        if landmarker is not None:
            landmarker.close()
            self.landmarker = None
