"""AsanaSenseAI Flask dashboard entry point."""
from collections import deque
from pathlib import Path
import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

from src.angle_analyzer import calculate_joint_angles
from src.feature_extraction import landmarks_to_features
from src.feedback_engine import compare_angles
from src.pose_classifier import PoseClassifier
from src.pose_detection import PoseDetector
from src.reference_pose import load_dataset, get_reference
from src.skeleton_alignment import align_reference

load_dotenv()
ROOT = Path(__file__).resolve().parent
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
DEFAULT_POSE = os.getenv("TARGET_POSE", "Bound_Angle_Pose")
POSE_NAMES = {
    "Bound_Angle_Pose": "Bound Angle Pose",
    "Cobra_Pose": "Cobra Pose",
    "Downdog_Pose": "Downward-Facing Dog",
    "Goddess_Pose": "Goddess Pose",
    "Plank_Pose": "Plank Pose",
    "Tree_Pose": "Tree Pose",
    "Warrior2_Pose": "Warrior II Pose",
}

app = Flask(__name__, template_folder="templates", static_folder="static")
state_lock = threading.Lock()
dataset = load_dataset()
classifier = PoseClassifier()
pose_detector = PoseDetector()
camera = None
camera_lock = threading.Lock()
stop_event = threading.Event()
state = {
    "accuracy": 0.0, "predicted_pose": "Waiting for camera", "ai_confidence": 0.0,
    "joints": {}, "suggestions": [], "pose_detected": False,
    "target_pose": DEFAULT_POSE, "history": [], "model_ready": classifier.ready,
    "dataset_ready": not dataset.empty,
}
target_pose = DEFAULT_POSE
reference_angles, ghost_points = get_reference(dataset, target_pose)


def available_poses():
    if not dataset.empty and "label" in dataset.columns:
        found = set(dataset["label"].dropna().astype(str))
        return [{"key": key, "name": name} for key, name in POSE_NAMES.items() if key in found]
    return [{"key": key, "name": name} for key, name in POSE_NAMES.items()]


def get_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(CAMERA_INDEX,cv2.CAP_DSHOW)
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        return camera


def draw_ghost(frame, points, landmarks):
    """Draw an adaptive white reference skeleton over the live camera feed."""

    if points is None or landmarks is None or len(points) < 33:
        return frame

    h, w = frame.shape[:2]

    try:
        aligned = np.asarray(
            align_reference(points, landmarks),
            dtype=np.float64,
        )
    except (ValueError, np.linalg.LinAlgError, TypeError, IndexError):
        return frame

    if aligned.ndim != 2 or aligned.shape[0] < 33 or aligned.shape[1] < 2:
        return frame

    xy = aligned[:33, :2]

    if not np.isfinite(xy).all():
        return frame

    connections = [
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 29), (29, 31),
        (24, 26), (26, 28), (28, 30), (30, 32),
        (15, 17), (15, 19), (15, 21),
        (16, 18), (16, 20), (16, 22),
    ]

    overlay = frame.copy()

    for start, end in connections:
        x1, y1 = xy[start]
        x2, y2 = xy[end]

        # Ignore unreasonable coordinates to avoid lines across the screen.
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 3:
            continue

        p1 = (round(x1 * w), round(y1 * h))
        p2 = (round(x2 * w), round(y2 * h))

        visible, p1, p2 = cv2.clipLine((0, 0, w, h), p1, p2)

        if visible:
            cv2.line(
                overlay,
                p1,
                p2,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

    # Draw visible reference joints.
    for x, y in xy:
        if abs(x) > 3 or abs(y) > 3:
            continue

        px, py = round(x * w), round(y * h)

        if 0 <= px < w and 0 <= py < h:
            cv2.circle(
                overlay,
                (px, py),
                4,
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )

    # Blend the white reference clearly over the original video.
    return cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)





def generate_frames():
    history_tick = time.time()
    ghost_start_time=time.time()
    while not stop_event.is_set():
        cap = get_camera()
        if cap is None or not cap.isOpened():
            placeholder = np.zeros((540, 960, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera unavailable - check webcam permissions", (35, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
            ok, encoded = cv2.imencode(".jpg", placeholder)
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
            time.sleep(0.5)
            continue
        success, frame = cap.read()
        if not success:
            time.sleep(0.05)
            continue
        frame = cv2.flip(frame, 1)
        result = pose_detector.detect(frame)
        detected, prediction, confidence = False, "No pose detected", 0.0
        joints, suggestions, accuracy = {}, [], 0.0
        if result and result.pose_landmarks:
            detected = True

            # Support both MediaPipe Tasks and Solutions result formats
            if isinstance(result.pose_landmarks, list):
                landmarks = result.pose_landmarks[0]
            else:
                landmarks = result.pose_landmarks.landmark
            try:
                prediction, confidence = classifier.predict(landmarks_to_features(landmarks))
            except Exception as exc:
                prediction, confidence = "Prediction unavailable", 0.0
                print(f"Prediction error: {exc}")
             
            live_angles = calculate_joint_angles(
                landmarks,
                frame_shape=frame.shape,
                visibility_threshold=0.65,
            )

            # Score only the joints that have reliable live measurements.
            accuracy, joints, suggestions = compare_angles(
                live_angles,
                reference_angles,
            )

            # Do not report a misleading score when too few joints are visible.
            if len(live_angles) < 3:
                accuracy = 0.0
                joints = {}
                suggestions = ["Move into full view of the camera."]

            if ghost_points and len(ghost_points) >= 33:
                frame = draw_ghost(frame, ghost_points, landmarks)
            pose_detector.draw_landmarks(frame, result)


        cv2.rectangle(frame, (8, 8), (390, 75), (20, 24, 34), -1)
        cv2.putText(frame, f"Target: {POSE_NAMES.get(target_pose, target_pose)}", (18, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 245), 1, cv2.LINE_AA)
        cv2.putText(frame, f"AI: {prediction}  |  Confidence: {confidence:.1f}%", (18, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 245, 170), 1, cv2.LINE_AA)
        with state_lock:
            state.update({"accuracy": accuracy, "predicted_pose": prediction,
                          "ai_confidence": round(confidence, 1), "joints": joints,
                          "suggestions": suggestions, "pose_detected": detected,
                          "target_pose": target_pose, "model_ready": classifier.ready,
                          "dataset_ready": not dataset.empty})
            if time.time() - history_tick >= 1:
                history_tick = time.time()
                state["history"].append({"t": int(time.time()), "acc": accuracy})
                state["history"] = state["history"][-60:]
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"


@app.route("/")
def index():
    return render_template("index.html", poses=available_poses(), target_pose=target_pose,
                           pose_names=POSE_NAMES)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/stats")
def stats():
    with state_lock:
        payload = dict(state)
        payload["joints"] = dict(state["joints"])
        payload["suggestions"] = list(state["suggestions"])
        payload["history"] = list(state["history"])
    payload["target_name"] = POSE_NAMES.get(target_pose, target_pose)
    return jsonify(payload)


@app.route("/set_target_pose", methods=["POST"])
def set_target_pose():
    global target_pose, reference_angles, ghost_points
    body = request.get_json(silent=True) or {}
    selected = body.get("pose", "")
    allowed = {item["key"] for item in available_poses()}
    if selected not in allowed:
        return jsonify({"success": False, "error": "Pose is not available in the current dataset."}), 400
    new_angles, new_points = get_reference(dataset, selected)
    if not new_angles or not new_points:
        return jsonify({"success": False, "error": "No valid reference samples found for this pose."}), 404
    with state_lock:
        target_pose = selected
        reference_angles, ghost_points = new_angles, new_points
        state.update({"target_pose": selected, "accuracy": 0.0, "joints": {}, "suggestions": [], "history": []})
    return jsonify({"success": True, "pose": selected, "name": POSE_NAMES.get(selected, selected)})


@app.route("/health")
def health():
    return jsonify({"ok": True, "model_ready": classifier.ready,
                    "dataset_ready": not dataset.empty,
                    "available_poses": [p["key"] for p in available_poses()]})


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    print(f"AsanaSenseAI dashboard: http://{host}:{port}")
    print("Model trained:", classifier.ready, "| Dataset loaded:", not dataset.empty)
    try:
        app.run(host=host, port=port, debug=False, threaded=True)
    finally:
        stop_event.set()
        with camera_lock:
            if camera is not None:
                camera.release()
        pose_detector.close()
