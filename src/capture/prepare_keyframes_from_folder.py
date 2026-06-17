from __future__ import annotations

import argparse
from pathlib import Path
import os

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def read_image_any(path: Path) -> np.ndarray | None:
    """Read common images with OpenCV; fallback to Pillow for HEIC/HEIF."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        return img

    if path.suffix.lower() not in {".heic", ".heif"}:
        return None

    try:
        from PIL import Image
    except Exception:
        return None

    try:
        with Image.open(path) as pil_img:
            rgb = np.array(pil_img.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize and quality-filter source images into reconstruction keyframes."
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=int(os.environ.get("SPECTRA_MAX_KEYFRAMES", "25")),
        help="Maximum number of prepared keyframes to keep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_source = os.environ.get("SPECTRA_FRAME_SOURCE", "raw_phone").strip().lower()
    if frame_source == "selected":
        src_dir = PROJECT_ROOT / "data" / "selected"
        source_label = "selected smart-frame output"
    else:
        src_dir = PROJECT_ROOT / "data" / "raw_phone"
        source_label = "raw phone images"
    out_dir = PROJECT_ROOT / "data" / "keyframes"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_w, target_h = 960, 720
    sharpness_thresh = 20.0
    max_images = max(1, args.max_images)

    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in exts])

    print("=== Keyframe Prep Starting ===", flush=True)
    print(f"Working directory : {Path.cwd().resolve()}", flush=True)
    print(f"Frame source      : {source_label}", flush=True)
    print(f"Source folder     : {src_dir.resolve()}", flush=True)
    print(f"Output folder     : {out_dir.resolve()}", flush=True)
    print(f"Found images      : {len(paths)}", flush=True)
    print(f"Max images        : {max_images}", flush=True)

    if not paths:
        raise FileNotFoundError(
            f"No images found in {src_dir}. "
            "Upload images from the dashboard, place images in data/raw_phone, "
            "or run Smart Frame Selection first."
        )

    removed = 0
    for p in out_dir.glob("*"):
        if p.is_file():
            p.unlink()
            removed += 1
    print(f"Cleared old keyframes: {removed}", flush=True)

    kept = 0
    rejected_blur = 0
    rejected_read = 0

    for index, p in enumerate(paths, start=1):
        if kept >= max_images:
            break

        print(f"[{index}/{len(paths)}] Reading {p.name}", flush=True)
        img = read_image_any(p)
        if img is None:
            rejected_read += 1
            print(f"  rejected unreadable: {p.name}", flush=True)
            continue

        img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        sharp = laplacian_sharpness(gray)

        if sharp < sharpness_thresh:
            rejected_blur += 1
            print(f"  rejected blurry: {p.name} sharpness={sharp:.2f}", flush=True)
            continue

        out_path = out_dir / f"frame_{kept:03d}.jpg"
        if cv2.imwrite(str(out_path), img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            print(f"  kept -> {out_path.name} sharpness={sharp:.2f}", flush=True)
            kept += 1
        else:
            rejected_read += 1
            print(f"  failed to write: {out_path}", flush=True)

    print("=== Keyframe Prep Report ===", flush=True)
    print(f"Source folder : {src_dir.resolve()}", flush=True)
    print(f"Output folder : {out_dir.resolve()}", flush=True)
    print(f"Kept          : {kept}", flush=True)
    print(f"Rejected (blurry)     : {rejected_blur}  (threshold={sharpness_thresh})", flush=True)
    print(f"Rejected (unreadable) : {rejected_read}", flush=True)
    if rejected_read > 0:
        print("Tip: install Pillow and pillow-heif if HEIC files fail to decode.", flush=True)
    print("Next: run sparse reconstruction -> python src/sfm/sparse_recon_live.py", flush=True)


if __name__ == "__main__":
    main()
