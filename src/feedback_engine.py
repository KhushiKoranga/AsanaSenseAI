"""Convert joint deviations into readable coaching hints."""
HINTS = {
    "left_shoulder": "Adjust your left shoulder to better match the reference.",
    "right_shoulder": "Adjust your right shoulder to better match the reference.",
    "left_elbow": "Check your left elbow bend.",
    "right_elbow": "Check your right elbow bend.",
    "left_hip": "Adjust your left hip position.",
    "right_hip": "Adjust your right hip position.",
    "left_knee": "Check your left knee bend.",
    "right_knee": "Check your right knee bend.",
    "left_ankle": "Check your left ankle and foot alignment.",
    "right_ankle": "Check your right ankle and foot alignment.",
}


def compare_angles(live_angles, reference_angles):
    joints, deviations = {}, []
    for name, angle in live_angles.items():
        if name not in reference_angles:
            continue
        # Absolute angle difference is more stable than dividing by a tiny reference angle.
        deviation = abs(float(angle) - float(reference_angles[name]))
        ok = deviation <= 15.0
        joints[name] = {"angle": round(float(angle), 1),
                        "deviation": round(deviation, 1), "ok": ok}
        deviations.append(deviation)
    # A heuristic similarity score, not a medically validated correctness score.
    score = max(0.0, 100.0 - (sum(deviations) / len(deviations))) if deviations else 0.0
    suggestions = [HINTS[name] for name, values in joints.items()
                   if not values["ok"] and name in HINTS]
    return round(score, 1), joints, suggestions[:3]
