import time
from pathlib import Path

import cv2
import numpy as np


def laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> None:
    out_dir = Path("data/keyframes")
    out_dir.mkdir(parents=True, exist_ok=True)

    cam_index = 0
    w, h = 640, 480
    save_every_sec = 0.7
    sharpness_thresh = 80.0
    max_frames = 25

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try cam_index=1 or check permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    print("Controls: SPACE=save | Q=quit")
    print(f"Auto-save every ~{save_every_sec}s if sharp; stop at {max_frames} frames.")

    saved = 0
    last_save = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = laplacian_sharpness(gray)
        now = time.time()

        auto_save = (now - last_save) >= save_every_sec and sharp >= sharpness_thresh

        hud = frame.copy()
        cv2.putText(hud, f"Saved: {saved}/{max_frames}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(hud, f"Sharpness: {sharp:.1f} (>= {sharpness_thresh})", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(hud, "SPACE=save | Q=quit", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Keyframe Capture", hud)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == ord(" ") or auto_save:
            fname = out_dir / f"frame_{saved:03d}.jpg"
            cv2.imwrite(str(fname), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            print(f"Saved {fname} | sharpness={sharp:.1f}")
            saved += 1
            last_save = now

        if saved >= max_frames:
            print("Reached max frames.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()