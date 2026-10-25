
import time
import cv2

from src.pose_detection import PoseDetector

def main():
    print("=" * 55)
    print("       ASANASENSEAI - REAL-TIME POSE DETECTION")
    print("=" * 55)

    detector = None
    camera = None

    try:
        print("[INFO] Loading pose detection model...")
        detector = PoseDetector()
        print("[SUCCESS] Pose detection model loaded.")

        print("[INFO] Opening webcam...")
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            raise RuntimeError(
                "Webcam could not be opened. Check camera permissions."
            )

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("[SUCCESS] Webcam connected.")
        print("[INFO] Press Q or ESC to exit.")

        previous_time = time.perf_counter()
        fps = 0.0

        while True:
            success, frame = camera.read()

            if not success or frame is None:
                print("[WARNING] Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)

            results = detector.detect(frame)
            detector.draw_landmarks(frame, results)

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time

            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = (
                    instant_fps
                    if fps == 0
                    else 0.9 * fps + 0.1 * instant_fps
                )

            pose_found = (
                results is not None
                and bool(results.pose_landmarks)
            )

            status = (
                "POSE DETECTED"
                if pose_found
                else "NO POSE DETECTED"
            )
            status_color = (
                (60, 220, 60)
                if pose_found
                else (0, 80, 255)
            )

            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (0, 0),
                (frame.shape[1], 100),
                (20, 20, 20),
                -1,
            )
            cv2.addWeighted(
                overlay, 0.65, frame, 0.35, 0, frame
            )

            cv2.putText(
                frame, "AsanaSenseAI", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA
            )

            cv2.putText(
                frame, status, (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                status_color, 2, cv2.LINE_AA
            )

            cv2.putText(
                frame, f"FPS: {fps:.1f}",
                (frame.shape[1] - 160, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA
            )

            cv2.putText(
                frame, "Press Q or ESC to quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA
            )

            cv2.imshow(
                "AsanaSenseAI - Live Pose Detection",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

    except Exception as error:
        print(f"\n[ERROR] {type(error).__name__}: {error}")

    finally:
        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()

        if detector is not None:
            detector.close()

        print("[INFO] Webcam resources released.")
        print("[INFO] AsanaSenseAI webcam test ended.")


if __name__ == "__main__":
    main()
