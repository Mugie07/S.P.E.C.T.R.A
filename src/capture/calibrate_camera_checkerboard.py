from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))


ARUCO_DICTS = {
    "DICT_4X4_50":  cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50":  cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50":  cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_7X7_50":  cv2.aruco.DICT_7X7_50,
}

# Reject any calibration image whose per-image reprojection error exceeds this.
# Images above this threshold are dragging your RMS up and hurting geometry.
PER_IMAGE_ERROR_THRESHOLD = 1.0  # pixels

# Minimum images that must survive filtering before we accept the calibration.
MIN_IMAGES_AFTER_FILTER = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate a camera from ChArUco or checkerboard photos and save "
            "intrinsics to JSON. Includes subpixel corner refinement and "
            "automatic per-image outlier rejection to minimise RMS error."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/calibration"),
        help="Folder containing calibration images (jpg/jpeg/png).",
    )
    parser.add_argument(
        "--board_type",
        choices=("checkerboard", "charuco"),
        default="charuco",
        help="Calibration board type present in the images.",
    )
    parser.add_argument(
        "--pattern_cols",
        type=int,
        required=True,
        help="ChArUco square count along width, or checkerboard inner corners along width.",
    )
    parser.add_argument(
        "--pattern_rows",
        type=int,
        required=True,
        help="ChArUco square count along height, or checkerboard inner corners along height.",
    )
    parser.add_argument(
        "--square_size",
        type=float,
        default=1.0,
        help="Physical size of one square in any consistent unit.",
    )
    parser.add_argument(
        "--marker_size",
        type=float,
        default=0.7,
        help="ChArUco ArUco marker size in the same unit as square_size.",
    )
    parser.add_argument(
        "--aruco_dict",
        choices=tuple(ARUCO_DICTS.keys()),
        default="DICT_6X6_100",
        help="ArUco dictionary used when board_type=charuco.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/camera_intrinsics.json"),
        help="Where to save the calibration JSON.",
    )
    return parser.parse_args()


def collect_image_paths(input_dir: Path) -> list[Path]:
    image_paths = sorted(
        list(input_dir.glob("*.jpg")) +
        list(input_dir.glob("*.jpeg")) +
        list(input_dir.glob("*.png"))
    )
    if not image_paths:
        raise FileNotFoundError(f"No calibration images found in {input_dir}")
    print(f"Found {len(image_paths)} images in {input_dir}")
    return image_paths


def build_charuco_board(args: argparse.Namespace):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[args.aruco_dict])
    board = cv2.aruco.CharucoBoard(
        (args.pattern_cols, args.pattern_rows),
        args.square_size,
        args.marker_size,
        dictionary,
    )
    return dictionary, board


# ---------------------------------------------------------------------------
# Checkerboard path
# ---------------------------------------------------------------------------

def calibrate_from_checkerboard(
    args: argparse.Namespace,
    image_paths: list[Path],
) -> tuple[float, np.ndarray, np.ndarray, tuple[int, int], int]:

    pattern_size = (args.pattern_cols, args.pattern_rows)
    objp = np.zeros((args.pattern_rows * args.pattern_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.pattern_cols, 0:args.pattern_rows].T.reshape(-1, 2)
    objp *= args.square_size

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for image_path in image_paths:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  Skipping unreadable: {image_path.name}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            print(f"  No board found: {image_path.name}")
            continue

        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        print(f"  Accepted: {image_path.name}")

    if len(objpoints) < 3:
        raise RuntimeError("Need at least 3 successful checkerboard images.")
    if image_size is None:
        raise RuntimeError("Could not determine image size.")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    print(f"\nInitial RMS: {rms:.4f}px  ({len(objpoints)} images)")

    # Per-image error filtering
    rms, K, dist, num_used = _filter_and_recalibrate_checkerboard(
        objpoints, imgpoints, image_size, rvecs, tvecs, K, dist
    )
    return rms, K, dist, image_size, num_used


def _filter_and_recalibrate_checkerboard(
    objpoints, imgpoints, image_size, rvecs, tvecs, K, dist
):
    per_image_errors = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        per_image_errors.append(err)

    _print_per_image_errors(per_image_errors)

    filtered_obj, filtered_img = [], []
    for i, err in enumerate(per_image_errors):
        if err <= PER_IMAGE_ERROR_THRESHOLD:
            filtered_obj.append(objpoints[i])
            filtered_img.append(imgpoints[i])
        else:
            print(f"  Rejecting image {i} — error {err:.4f}px > threshold {PER_IMAGE_ERROR_THRESHOLD}px")

    if len(filtered_obj) < MIN_IMAGES_AFTER_FILTER:
        print(
            f"Warning: only {len(filtered_obj)} images survived filtering "
            f"(minimum {MIN_IMAGES_AFTER_FILTER}). Keeping original calibration."
        )
        return cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)[0], K, dist, len(objpoints)

    rms, K, dist, _, _ = cv2.calibrateCamera(
        filtered_obj, filtered_img, image_size, None, None
    )
    print(f"Refined RMS: {rms:.4f}px  ({len(filtered_obj)} images after filtering)")
    return rms, K, dist, len(filtered_obj)


# ---------------------------------------------------------------------------
# ChArUco path
# ---------------------------------------------------------------------------

def calibrate_from_charuco(
    args: argparse.Namespace,
    image_paths: list[Path],
) -> tuple[float, np.ndarray, np.ndarray, tuple[int, int], int]:

    dictionary, board = build_charuco_board(args)

    # OpenCV 4.7+ uses ArucoDetector object instead of detectMarkers function
    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    # CharucoDetector for the new API
    charuco_detector = cv2.aruco.CharucoDetector(board)

    all_charuco_corners: list[np.ndarray] = []
    all_charuco_ids: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    accepted = 0

    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print("\n--- Detecting ChArUco corners ---")
    for image_path in image_paths:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  Skipping unreadable: {image_path.name}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])

        corners, ids, _ = aruco_detector.detectMarkers(gray)
        marker_count = 0 if ids is None else len(ids)

        if ids is None or marker_count < 4:
            print(f"  Rejected (too few markers={marker_count}): {image_path.name}")
            continue

        # OpenCV 4.7+ CharucoDetector API replaces interpolateCornersCharuco
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
        charuco_count = 0 if charuco_ids is None else len(charuco_ids)

        if charuco_corners is None or charuco_ids is None or charuco_count < 6:
            print(f"  Rejected (too few corners={charuco_count}): {image_path.name}")
            continue

        # Subpixel refinement — biggest single improvement to RMS quality
        charuco_corners = cv2.cornerSubPix(
            gray,
            charuco_corners,
            winSize=(5, 5),
            zeroZone=(-1, -1),
            criteria=subpix_criteria,
        )

        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        accepted += 1
        print(f"  Accepted: {image_path.name}  markers={marker_count}  corners={charuco_count}")

    if accepted < 3:
        raise RuntimeError(
            f"Need at least 3 successful ChArUco images, got {accepted}. "
            "Check that your board is fully visible in every frame and the "
            "ArUco dictionary matches the one used to print the board."
        )
    if image_size is None:
        raise RuntimeError("Could not determine image size.")

    print(f"\n--- Initial calibration from {accepted} images ---")

    # Convert charuco corners to object/image point pairs for cv2.calibrateCamera
    # This works across all OpenCV 4.x versions
    obj_points_all, img_points_all = _charuco_to_calib_points(
        all_charuco_corners, all_charuco_ids, board
    )

    if len(obj_points_all) < 3:
        raise RuntimeError("Not enough valid point pairs for calibration.")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points_all, img_points_all, image_size, None, None
    )
    print(f"Initial RMS: {rms:.4f}px")

    # Per-image error filtering and recalibration
    rms, K, dist, num_used = _filter_and_recalibrate_charuco(
        obj_points_all, img_points_all, image_size, rvecs, tvecs, K, dist
    )
    return rms, K, dist, image_size, num_used


def _charuco_to_calib_points(
    all_corners: list, all_ids: list, board
) -> tuple[list, list]:
    """
    Convert ChArUco corner detections to object/image point pairs
    compatible with cv2.calibrateCamera. Works on all OpenCV 4.x versions.
    """
    obj_points_all = []
    img_points_all = []
    for corners, ids in zip(all_corners, all_ids):
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        if obj_pts is not None and len(obj_pts) >= 4:
            obj_points_all.append(obj_pts.astype(np.float32))
            img_points_all.append(img_pts.astype(np.float32))
    return obj_points_all, img_points_all


def _filter_and_recalibrate_charuco(
    obj_points_all, img_points_all, image_size, rvecs, tvecs, K, dist
):
    """
    Compute per-image reprojection errors, reject outliers above threshold,
    and recalibrate on the surviving images.
    """
    per_image_errors = []
    for i in range(len(obj_points_all)):
        projected, _ = cv2.projectPoints(
            obj_points_all[i].astype(np.float64),
            rvecs[i], tvecs[i],
            K, dist,
        )
        projected = projected.astype(np.float32)
        err = cv2.norm(img_points_all[i], projected.reshape(-1, 1, 2), cv2.NORM_L2) / len(projected)
        per_image_errors.append(float(err))

    _print_per_image_errors(per_image_errors)

    filtered_obj, filtered_img = [], []
    for i, err in enumerate(per_image_errors):
        if err <= PER_IMAGE_ERROR_THRESHOLD:
            filtered_obj.append(obj_points_all[i])
            filtered_img.append(img_points_all[i])
        else:
            print(f"  Rejecting image {i} — error {err:.4f}px > threshold {PER_IMAGE_ERROR_THRESHOLD}px")

    if len(filtered_obj) < MIN_IMAGES_AFTER_FILTER:
        print(
            f"Warning: only {len(filtered_obj)} images survived filtering "
            f"(minimum {MIN_IMAGES_AFTER_FILTER}). Keeping original calibration.\n"
            f"To fix this: recapture calibration images with better coverage, "
            f"no motion blur, and the full board always visible."
        )
        return float(np.mean(per_image_errors)), K, dist, len(obj_points_all)

    rms, K_new, dist_new, _, _ = cv2.calibrateCamera(
        filtered_obj, filtered_img, image_size, None, None
    )
    print(
        f"\nRefined RMS: {rms:.4f}px  "
        f"({len(filtered_obj)} images kept, "
        f"{len(obj_points_all) - len(filtered_obj)} rejected)"
    )
    return rms, K_new, dist_new, len(filtered_obj)


def _print_per_image_errors(errors: list[float]) -> None:
    print("\n--- Per-image reprojection errors ---")
    for i, err in enumerate(errors):
        status = "OK" if err <= PER_IMAGE_ERROR_THRESHOLD else "REJECT"
        bar = "#" * min(int(err * 20), 40)
        print(f"  [{i:02d}] {err:6.3f}px  {bar}  {status}")
    print()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_calibration_diagnostics(
    K: np.ndarray,
    dist: np.ndarray,
    rms: float,
    image_size: tuple[int, int],
) -> None:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    w, h = image_size
    asymmetry_pct = abs(fx - fy) / max(fx, fy) * 100.0

    print("\n" + "="*55)
    print("CALIBRATION RESULT")
    print("="*55)
    print(f"  RMS reprojection error : {rms:.4f}px", end="")
    if rms < 0.5:
        print("  ✓ Excellent")
    elif rms < 1.0:
        print("  ✓ Good")
    elif rms < 2.0:
        print("  ⚠ Acceptable but recheck your capture")
    else:
        print("  ✗ Poor — recalibrate with better images")

    print(f"  fx / fy                : {fx:.1f} / {fy:.1f}", end="")
    if asymmetry_pct < 2.0:
        print(f"  ✓ ({asymmetry_pct:.1f}% asymmetry)")
    elif asymmetry_pct < 5.0:
        print(f"  ⚠ ({asymmetry_pct:.1f}% asymmetry — acceptable)")
    else:
        print(f"  ✗ ({asymmetry_pct:.1f}% asymmetry — bad calibration)")

    print(f"  Principal point cx/cy  : {cx:.1f} / {cy:.1f}  (image centre: {w/2:.1f} / {h/2:.1f})")

    cx_offset_pct = abs(cx - w / 2) / w * 100
    cy_offset_pct = abs(cy - h / 2) / h * 100
    if cx_offset_pct < 5 and cy_offset_pct < 5:
        print(f"  Principal point offset : ✓ Near centre")
    else:
        print(f"  Principal point offset : ⚠ {cx_offset_pct:.1f}% / {cy_offset_pct:.1f}% from centre")

    # FIX: flatten dist from shape (1,5) or (5,1) to a 1-D array before indexing
    d = dist.flatten()
    k1, k2 = float(d[0]), float(d[1])
    k3 = float(d[4]) if len(d) > 4 else 0.0

    print(f"  Distortion k1/k2/k3   : {k1:.4f} / {k2:.4f} / {k3:.4f}", end="")
    if abs(k3) > 1.0:
        print("  ⚠ k3 is large — possible overfitting")
    else:
        print("  ✓")

    print("="*55)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    image_paths = collect_image_paths(args.input_dir)

    if args.board_type == "checkerboard":
        rms, K, dist, image_size, num_images_used = calibrate_from_checkerboard(args, image_paths)
    else:
        rms, K, dist, image_size, num_images_used = calibrate_from_charuco(args, image_paths)

    print_calibration_diagnostics(K, dist, rms, image_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "board_type": args.board_type,
        "width":  image_size[0],
        "height": image_size[1],
        "K": K.tolist(),
        "dist_coeffs": dist.reshape(-1).tolist(),
        "rms_reprojection_error": float(rms),
        "pattern_cols": args.pattern_cols,
        "pattern_rows": args.pattern_rows,
        "square_size": args.square_size,
        "num_images_used": int(num_images_used),
    }
    if args.board_type == "charuco":
        payload["aruco_dict"] = args.aruco_dict
        payload["marker_size"] = args.marker_size

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved calibration to : {args.output}")
    print(f"RMS reprojection error: {rms:.4f}px  (used {num_images_used} images)")


if __name__ == "__main__":
    main()