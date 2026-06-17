"""
smart_frame_selector.py
-----------------------
Unified entry point for frame selection supporting two input modes:

  MODE 1 - Still images (original workflow):
      python src/utils/smart_frame_selector.py --mode images

  MODE 2 - Video extraction (new workflow):
      python src/utils/smart_frame_selector.py --mode video --video_path data/video/capture.mp4

In both modes the output is the same: up to MAX_FRAMES diverse, sharp,
well-overlapping frames saved to data/selected/ ready for the pipeline.

The smart selector uses three filters:
  1. ORB feature count     — rejects blurry or textureless frames
  2. Perceptual hash       — rejects near-duplicate viewpoints
  3. Histogram difference  — rejects same-exposure redundant frames

Frames are scored by sharpness + feature richness so the best quality
frames are always preferred over mediocre ones when diversity is equal.

Full pipeline after running this script:
  python src/capture/prepare_keyframes_from_folder.py  (reads data/selected/)
  python src/depth/midas_depth.py
  python src/sfm/sparse_recon_live.py --no_view
  python src/fusion/depth_fusion.py --no_view --remove_plane
  python src/utils/cleanup.py
  python src/mesh/mesh_surface.py
  python src/utils/evaluate_reconstruction.py
  python src/export/gaussian_export.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Still images input directory (MODE 1)
IMAGES_INPUT_DIR = Path("data/raw_phone")

# Video input directory (MODE 2) — can be overridden by --video_path
VIDEO_INPUT_DIR  = Path("data/video")

# Output directory — same for both modes, feeds into existing pipeline
OUTPUT_DIR = Path("data/selected")


# ---------------------------------------------------------------------------
# Tuning parameters
# ---------------------------------------------------------------------------

# Maximum frames to keep after filtering.
# Increased from 25 to 30 as requested — more diverse frames = better coverage.
MAX_FRAMES = 24

# Minimum perceptual hash distance between kept frames (0-64).
# Higher = frames must look more different to both be kept.
# 8  = light filtering (many similar frames allowed)
# 10 = moderate filtering — recommended default
# 18 = aggressive filtering (only very different frames kept)
MIN_HASH_DISTANCE = 14

# Minimum histogram difference between kept frames (0.0-1.0).
# Catches exposure/lighting changes that hash distance alone misses.
MIN_HIST_DIFF = 0.04

# Minimum ORB feature points a frame must have to be considered usable.
# Frames below this threshold are blurry or featureless — bad for SfM.
MIN_FEATURE_POINTS = 300

# Resize images to this for hash/histogram comparison (speed optimisation).
COMPARE_SIZE = (256, 256)

# ---------------------------------------------------------------------------
# Video extraction parameters (MODE 2 only)
# ---------------------------------------------------------------------------

# Extract one frame every N frames from the video.
# At 30fps: EXTRACT_EVERY=10 gives 3 frames/second of walking.
# At 60fps: EXTRACT_EVERY=20 gives 3 frames/second of walking.
# Increase if you are getting too many near-identical frames.
# Decrease if coverage feels sparse after selection.
EXTRACT_EVERY = 10


# ---------------------------------------------------------------------------
# Perceptual hash utilities
# ---------------------------------------------------------------------------

def phash(img: np.ndarray, hash_size: int = 16) -> np.ndarray:
    """
    Compute a perceptual hash of an image.
    Similar images produce similar hashes regardless of minor colour or
    exposure differences. Used to detect near-duplicate frames.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = resized.mean()
    return (resized > mean).flatten()


def hash_distance(h1: np.ndarray, h2: np.ndarray) -> int:
    """Hamming distance between two perceptual hashes. Lower = more similar."""
    return int(np.count_nonzero(h1 != h2))


def histogram_diff(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Normalised histogram difference between two images.
    Returns 0.0 for identical histograms, higher for more different images.
    """
    h1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    h2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))


def count_features(img: np.ndarray) -> int:
    """
    Count ORB feature keypoints in an image.
    Frames with few features are blurry or textureless — bad for SfM.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints = orb.detect(gray, None)
    return len(keypoints)


def read_image_any(path: Path) -> np.ndarray | None:
    """Read common reconstruction image formats; use Pillow fallback for HEIC/HEIF."""
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


def score_frame(img: np.ndarray) -> float:
    """
    Score a frame by reconstruction usefulness.
    Higher score = sharper + richer in features = preferred during selection.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    features = count_features(img)
    return sharpness * 0.7 + features * 0.3


# ---------------------------------------------------------------------------
# Mode 1 — Load still images
# ---------------------------------------------------------------------------

def load_images_from_folder(input_dir: Path) -> list[tuple[Path, np.ndarray]]:
    """Load all JPG/PNG images from the raw_phone folder."""
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}\n"
            "Place your captured images in data/raw_phone/"
        )

    paths = sorted([path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES])

    if not paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    print(f"[Mode: images] Found {len(paths)} images in {input_dir}")

    frames = []
    for path in paths:
        img = read_image_any(path)
        if img is None:
            print(f"  Skipping unreadable: {path.name}")
            continue
        frames.append((path, img))

    return frames


# ---------------------------------------------------------------------------
# Mode 2 — Extract frames from video
# ---------------------------------------------------------------------------

def extract_frames_from_video(
    video_path: Path,
) -> list[tuple[Path, np.ndarray]]:
    """
    Extract frames from a video file at regular intervals.
    Returns list of (synthetic_path, frame_image) tuples.
    The synthetic path is used for naming the output files only.
    """
    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file not found: {video_path}\n"
            "Place your video at data/video/capture.mp4 or pass --video_path"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    print(f"[Mode: video] {video_path.name}")
    print(f"  Total frames : {total_frames}")
    print(f"  FPS          : {fps:.1f}")
    print(f"  Duration     : {duration:.1f}s")
    print(f"  Extracting every {EXTRACT_EVERY} frames "
          f"(~{fps / EXTRACT_EVERY:.1f} frames/sec of footage)")

    frames = []
    frame_idx = 0
    extracted = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % EXTRACT_EVERY == 0:
            synthetic_path = Path(f"video_frame_{extracted:04d}.jpg")
            frames.append((synthetic_path, frame))
            extracted += 1
        frame_idx += 1

    cap.release()
    print(f"  Extracted {extracted} candidate frames from video")
    return frames


# ---------------------------------------------------------------------------
# Smart selection (shared by both modes)
# ---------------------------------------------------------------------------

def select_diverse_frames(
    frames: list[tuple[Path, np.ndarray]],
    max_frames: int = MAX_FRAMES,
) -> list[tuple[Path, np.ndarray]]:
    """
    Select up to MAX_FRAMES images that are:
    1. Sharp and feature-rich enough to be useful for SfM
    2. Sufficiently different from all already-selected frames
    3. Ranked by quality so the best frames are always preferred
    """
    print(f"\nAnalysing {len(frames)} candidate frames...")

    # Score and pre-filter all frames
    candidates = []
    for order, (path, img) in enumerate(frames):
        features = count_features(img)
        if features < MIN_FEATURE_POINTS:
            print(f"  Rejected (too few features={features}): {path.name}")
            continue
        score = score_frame(img)
        small = cv2.resize(img, COMPARE_SIZE)
        h = phash(small)
        candidates.append((score, order, path, img, small, h))

    # Sort by score descending — best frames first
    candidates.sort(key=lambda x: x[0], reverse=True)
    print(f"  {len(candidates)} frames passed feature threshold "
          f"(rejected {len(frames) - len(candidates)} blurry/featureless)")

    # Greedy diverse selection
    selected: list[tuple] = []
    for score, order, path, img, small, h in candidates:
        if len(selected) >= max_frames:
            break

        # Check against all already-selected frames
        too_similar = False
        for _, _, _, _, sel_small, sel_h in selected:
            if hash_distance(h, sel_h) < MIN_HASH_DISTANCE:
                too_similar = True
                break
            if histogram_diff(small, sel_small) < MIN_HIST_DIFF:
                too_similar = True
                break

        if too_similar:
            continue

        selected.append((score, order, path, img, small, h))
        print(f"  Selected [{len(selected):02d}/{max_frames}] "
              f"score={score:.1f}: {path.name}")

    selected.sort(key=lambda item: item[1])
    return [(path, img) for _, _, path, img, _, _ in selected]


# ---------------------------------------------------------------------------
# Save selected frames to output directory
# ---------------------------------------------------------------------------

def save_selected(selected: list[tuple[Path, np.ndarray]]) -> None:
    """Save selected frames to OUTPUT_DIR, clearing any previous selection."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clear previous selection
    for f in OUTPUT_DIR.glob("*"):
        f.unlink()

    for i, (path, img) in enumerate(selected):
        dest = OUTPUT_DIR / f"sel_{i:03d}{path.suffix if path.suffix else '.jpg'}"
        cv2.imwrite(
            str(dest),
            img,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

    print(f"\n{'='*50}")
    print(f"Selected {len(selected)} diverse frames")
    print(f"Saved to : {OUTPUT_DIR}")
    print(f"{'='*50}")

    if len(selected) < 8:
        print(f"\nWarning: only {len(selected)} frames selected.")
        print("Capture more diverse viewpoints for reliable reconstruction.")
    else:
        print("\nNext step:")
        print("  python src/capture/prepare_keyframes_from_folder.py")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smart frame selector — filters the best diverse frames from "
            "still images or a video for 3D reconstruction."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("images", "video"),
        default="images",
        help=(
            "Input mode. "
            "'images' reads from data/raw_phone/ (original workflow). "
            "'video' extracts frames from a video file (new workflow)."
        ),
    )
    parser.add_argument(
        "--video_path",
        type=Path,
        default=VIDEO_INPUT_DIR / "capture.mp4",
        help=(
            "Path to video file when --mode video is used. "
            "Default: data/video/capture.mp4"
        ),
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=IMAGES_INPUT_DIR,
        help=(
            "Path to still images folder when --mode images is used. "
            "Default: data/raw_phone/"
        ),
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=MAX_FRAMES,
        help="Maximum number of selected frames to save.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Load frames from the appropriate source
    if args.mode == "images":
        frames = load_images_from_folder(args.images_dir)
    else:
        frames = extract_frames_from_video(args.video_path)

    # Run smart selection (identical for both modes)
    selected = select_diverse_frames(frames, max_frames=args.max_frames)

    # Save to output directory
    save_selected(selected)


if __name__ == "__main__":
    main()
