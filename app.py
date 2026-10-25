"""AsanaSenseAI Flask dashboard with browser webcam processing."""
import base64
import os
import threading
import time
import cv2
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
DEFAULT_POSE = os.getenv("TARGET_POSE", "Bound_Angle_Pose")
MAX_FRAME_BYTES = 2_000_000
POSE_NAMES = {
    "Bound_Angle_Pose": "Bound Angle Pose", "Cobra_Pose": "Cobra Pose",
    "Downdog_Pose": "Downward-Facing Dog", "Goddess_Pose": "Goddess Pose",
    "Plank_Pose": "Plank Pose", "Tree_Pose": "Tree Pose",
    "Warrior2_Pose": "Warrior II Pose",
}
app = Flask(__name__, template_folder="templates", static_folder="static")
state_lock = threading.Lock()
processing_lock = threading.Lock()
dataset = load_dataset()
classifier = PoseClassifier()
pose_detector = PoseDetector()
state = {
    "accuracy": 0.0, "predicted_pose": "Waiting for camera", "ai_confidence": 0.0,
    "joints": {}, "suggestions": ["Start your camera to begin."], "pose_detected": False,
    "target_pose": DEFAULT_POSE, "history": [], "model_ready": classifier.ready,
    "dataset_ready": not dataset.empty,
}
target_pose = DEFAULT_POSE
reference_angles, ghost_points = get_reference(dataset, target_pose)
history_tick = time.time()


def available_poses():
    if not dataset.empty and "label" in dataset.columns:
        found = set(dataset["label"].dropna().astype(str))
        return [{"key": k, "name": n} for k, n in POSE_NAMES.items() if k in found]
    return [{"key": k, "name": n} for k, n in POSE_NAMES.items()]


def draw_ghost(frame, points, landmarks):
    if points is None or landmarks is None or len(points) < 33:
        return frame
    h, w = frame.shape[:2]
    try:
        aligned = np.asarray(align_reference(points, landmarks), dtype=np.float64)
    except (ValueError, np.linalg.LinAlgError, TypeError, IndexError):
        return frame
    if aligned.ndim != 2 or aligned.shape[0] < 33 or aligned.shape[1] < 2:
        return frame
    xy = aligned[:33, :2]
    if not np.isfinite(xy).all():
        return frame
    links = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),
             (23,24),(23,25),(25,27),(27,29),(29,31),(24,26),(26,28),
             (28,30),(30,32),(15,17),(15,19),(15,21),(16,18),(16,20),(16,22)]
    overlay = frame.copy()
    for a, b in links:
        x1, y1 = xy[a]; x2, y2 = xy[b]
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 3:
            continue
        p1, p2 = (round(x1*w), round(y1*h)), (round(x2*w), round(y2*h))
        visible, p1, p2 = cv2.clipLine((0, 0, w, h), p1, p2)
        if visible:
            cv2.line(overlay, p1, p2, (255,255,255), 3, cv2.LINE_AA)
    for x, y in xy:
        if abs(x) <= 3 and abs(y) <= 3:
            px, py = round(x*w), round(y*h)
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(overlay, (px,py), 4, (255,255,255), -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)


def analyze_frame(frame):
    global history_tick
    frame = cv2.flip(frame, 1)
    result = pose_detector.detect(frame)
    detected, prediction, confidence = False, "No pose detected", 0.0
    joints, suggestions, accuracy = {}, [], 0.0
    if result and result.pose_landmarks:
        detected = True
        landmarks = result.pose_landmarks[0]
        try:
            prediction, confidence = classifier.predict(landmarks_to_features(landmarks))
        except Exception:
            app.logger.exception("Pose prediction failed")
            prediction, confidence = "Prediction unavailable", 0.0
        angles = calculate_joint_angles(landmarks, frame_shape=frame.shape, visibility_threshold=0.65)
        accuracy, joints, suggestions = compare_angles(angles, reference_angles)
        if len(angles) < 3:
            accuracy, joints = 0.0, {}
            suggestions = ["Move into full view of the camera."]
        if ghost_points and len(ghost_points) >= 33:
            frame = draw_ghost(frame, ghost_points, landmarks)
        pose_detector.draw_landmarks(frame, result)
    else:
        suggestions = ["Move back until your full body is visible."]
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (8, 8), (min(w-8, 510), 76), (20,24,34), -1)
    cv2.putText(frame, f"Target: {POSE_NAMES.get(target_pose, target_pose)}", (18,31),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (235,235,245), 1, cv2.LINE_AA)
    cv2.putText(frame, f"AI: {prediction} | Confidence: {confidence:.1f}%", (18,56),
                cv2.FONT_HERSHEY_SIMPLEX, .48, (120,245,170), 1, cv2.LINE_AA)
    with state_lock:
        state.update({"accuracy": round(float(accuracy),2), "predicted_pose": str(prediction),
                      "ai_confidence": round(float(confidence),1), "joints": joints,
                      "suggestions": suggestions, "pose_detected": detected,
                      "target_pose": target_pose, "model_ready": classifier.ready,
                      "dataset_ready": not dataset.empty})
        now = time.time()
        if now - history_tick >= 1:
            history_tick = now
            state["history"].append({"t": int(now), "acc": float(accuracy)})
            state["history"] = state["history"][-60:]
    return frame


@app.route("/")
def index():
    return render_template("index.html", poses=available_poses(), target_pose=target_pose,
                           pose_names=POSE_NAMES)


@app.route("/process_frame", methods=["POST"])
def process_frame():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image", "")
    if not isinstance(image_data, str) or "," not in image_data:
        return jsonify({"success": False, "error": "No valid camera frame received."}), 400
    try:
        raw = base64.b64decode(image_data.split(",", 1)[1], validate=True)
        if not raw or len(raw) > MAX_FRAME_BYTES:
            return jsonify({"success": False, "error": "Frame is empty or too large."}), 413
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"success": False, "error": "Could not decode camera frame."}), 400
        with processing_lock:
            annotated = analyze_frame(frame)
        ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            return jsonify({"success": False, "error": "Could not encode processed frame."}), 500
        with state_lock:
            result = dict(state)
            result["joints"] = dict(state["joints"])
            result["suggestions"] = list(state["suggestions"])
            result["history"] = list(state["history"])
        result.update({"success": True, "target_name": POSE_NAMES.get(target_pose, target_pose),
                       "frame": base64.b64encode(jpg.tobytes()).decode("ascii")})
        return jsonify(result)
    except (ValueError, TypeError) as exc:
        return jsonify({"success": False, "error": f"Invalid frame: {exc}"}), 400


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
    if selected not in {p["key"] for p in available_poses()}:
        return jsonify({"success": False, "error": "Pose is not available in the current dataset."}), 400
    new_angles, new_points = get_reference(dataset, selected)
    if not new_angles or not new_points:
        return jsonify({"success": False, "error": "No valid reference samples found for this pose."}), 404
    with state_lock:
        target_pose, reference_angles, ghost_points = selected, new_angles, new_points
        state.update({"target_pose": selected, "accuracy": 0.0, "joints": {},
                      "suggestions": [], "history": []})
    return jsonify({"success": True, "pose": selected, "name": POSE_NAMES.get(selected, selected)})


@app.route("/health")
def health():
    return jsonify({"ok": True, "model_ready": classifier.ready, "dataset_ready": not dataset.empty,
                    "available_poses": [p["key"] for p in available_poses()]})


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    print(f"AsanaSenseAI dashboard: http://{host}:{port}")
    print("Model trained:", classifier.ready, "| Dataset loaded:", not dataset.empty)
    try:
        app.run(host=host, port=port, debug=False, threaded=True)
    finally:
        pose_detector.close()
