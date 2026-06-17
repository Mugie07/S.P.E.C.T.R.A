from __future__ import annotations

import warnings
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# CHANGED: dual calibration paths for 1x and 2x zoom.
# Old behaviour: system only looked for one file — data/camera_intrinsics.json
# New behaviour: system looks for zoom-specific files first, then falls back
# to the generic file for backwards compatibility with existing setups.
# ---------------------------------------------------------------------------

CALIBRATION_PATHS = {
    "1x": [
        Path("data/camera_intrinsics_1x.json"),
        Path("data/results/camera_intrinsics_1x.json"),
    ],
    "2x": [
        Path("data/camera_intrinsics_2x.json"),
        Path("data/results/camera_intrinsics_2x.json"),
    ],
}

# Fallback paths used when no zoom is specified — backwards compatible
# with any existing scripts that don't pass a zoom argument yet.
DEFAULT_CALIBRATION_PATHS = (
    Path("data/camera_intrinsics.json"),
    Path("data/results/camera_intrinsics.json"),
)


@dataclass
class CameraCalibration:
    K: np.ndarray
    dist_coeffs: np.ndarray
    width: int
    height: int
    source: str


def make_default_intrinsics(width: int, height: int) -> tuple[float, float, float, float]:
    """Shared fallback intrinsics used across sparse and dense reconstruction."""
    f = 0.9 * max(width, height)
    cx = width / 2.0
    cy = height / 2.0
    return float(f), float(f), float(cx), float(cy)


def make_default_calibration(width: int, height: int) -> CameraCalibration:
    fx, fy, cx, cy = make_default_intrinsics(width, height)
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_coeffs = np.zeros((5,), dtype=np.float64)
    return CameraCalibration(
        K=K,
        dist_coeffs=dist_coeffs,
        width=width,
        height=height,
        source="default_fallback",
    )


def _rotate_intrinsics_90ccw(K: np.ndarray, width: int, height: int) -> np.ndarray:
    """Rotate intrinsics from a landscape image into portrait coordinates."""
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    rotated = np.array([
        [fy, 0.0, cy],
        [0.0, fx, width - 1.0 - cx],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return rotated


def adapt_calibration_to_image(
    calibration: CameraCalibration,
    target_width: int,
    target_height: int,
    aspect_tolerance: float = 0.03,
) -> CameraCalibration:
    """
    Adapt calibration captured at another resolution/orientation to a target image size.

    Supports:
    - Direct resize when aspect ratios match.
    - 90-degree rotation plus resize when width/height were swapped during capture.
    - Falls back to synthetic intrinsics with a loud warning if neither matches.
    """
    src_w = float(calibration.width)
    src_h = float(calibration.height)
    dst_w = float(target_width)
    dst_h = float(target_height)

    src_aspect = src_w / src_h
    dst_aspect = dst_w / dst_h
    source_label = calibration.source

    # --- Direct resize ---
    if abs(src_aspect - dst_aspect) <= aspect_tolerance:
        K = calibration.K.copy()
        K[0, 0] *= dst_w / src_w
        K[1, 1] *= dst_h / src_h
        K[0, 2] *= dst_w / src_w
        K[1, 2] *= dst_h / src_h
        return CameraCalibration(
            K=K,
            dist_coeffs=calibration.dist_coeffs.copy(),
            width=target_width,
            height=target_height,
            source=f"{source_label} [rescaled]",
        )

    # --- Rotation + resize (portrait <-> landscape swap) ---
    rotated_aspect = src_h / src_w
    if abs(rotated_aspect - dst_aspect) <= aspect_tolerance:
        K_rot = _rotate_intrinsics_90ccw(calibration.K, calibration.width, calibration.height)
        K_rot[0, 0] *= dst_w / src_h
        K_rot[1, 1] *= dst_h / src_w
        K_rot[0, 2] *= dst_w / src_h
        K_rot[1, 2] *= dst_h / src_w
        return CameraCalibration(
            K=K_rot,
            dist_coeffs=calibration.dist_coeffs.copy(),
            width=target_width,
            height=target_height,
            source=f"{source_label} [rotated+rescaled]",
        )

    # --- Fallback: calibration does not match — warn loudly ---
    warnings.warn(
        f"\n{'='*60}\n"
        f"CALIBRATION MISMATCH DETECTED\n"
        f"Source calibration: {source_label}\n"
        f"  Calibrated at: {int(src_w)}x{int(src_h)} (aspect={src_aspect:.4f})\n"
        f"  Target image:  {target_width}x{target_height} (aspect={dst_aspect:.4f})\n"
        f"Neither direct resize nor 90-degree rotation produced a matching aspect ratio.\n"
        f"Falling back to SYNTHETIC intrinsics. Reconstruction quality will be\n"
        f"significantly degraded. Recalibrate at the correct resolution/orientation.\n"
        f"{'='*60}",
        stacklevel=3,
    )
    fallback = make_default_calibration(target_width, target_height)
    fallback.source = f"default_fallback [calibration mismatch: {source_label}]"
    return fallback


# ---------------------------------------------------------------------------
# CHANGED: _find_calibration_file now accepts an optional zoom argument.
# Old behaviour: always searched DEFAULT_CALIBRATION_PATHS regardless of zoom.
# New behaviour: if zoom is "1x" or "2x", searches zoom-specific paths first,
# then falls back to the generic path for backwards compatibility.
# ---------------------------------------------------------------------------

def _find_calibration_file(
    calibration_path: Path | None = None,
    zoom: str | None = None,
) -> Path | None:
    """
    Find the best available calibration file.

    Search order:
    1. Explicit path if provided
    2. Zoom-specific path if zoom is "1x" or "2x"
    3. Generic fallback paths (backwards compatible)
    """
    # Explicit path always wins
    if calibration_path is not None:
        if calibration_path.exists():
            return calibration_path
        return None

    # Zoom-specific search
    if zoom in CALIBRATION_PATHS:
        for candidate in CALIBRATION_PATHS[zoom]:
            if candidate.exists():
                return candidate
        # Zoom file not found — warn before falling through to generic
        warnings.warn(
            f"[Calibration] No {zoom} calibration file found. "
            f"Expected one of: {[str(p) for p in CALIBRATION_PATHS[zoom]]}. "
            f"Falling back to generic calibration file. "
            f"Run calibrate_camera_checkerboard.py at {zoom} zoom and save "
            f"the result as data/camera_intrinsics_{zoom}.json.",
            stacklevel=3,
        )

    # Generic fallback — backwards compatible with existing single-file setups
    for candidate in DEFAULT_CALIBRATION_PATHS:
        if candidate.exists():
            return candidate

    return None


# ---------------------------------------------------------------------------
# CHANGED: load_camera_calibration now accepts an optional zoom argument.
# Old behaviour: no zoom awareness — always loaded the single generic file.
# New behaviour: pass zoom="1x" for room reconstruction or zoom="2x" for
# object reconstruction to load the correct lens calibration automatically.
# If zoom is None, behaviour is identical to the original (backwards compatible).
# ---------------------------------------------------------------------------

def load_camera_calibration(
    width: int,
    height: int,
    calibration_path: Path | None = None,
    zoom: str | None = None,
) -> CameraCalibration:
    """
    Load camera intrinsics and distortion coefficients from JSON if available.

    Args:
        width:            Target image width in pixels.
        height:           Target image height in pixels.
        calibration_path: Explicit path to a calibration JSON (overrides zoom).
        zoom:             "1x" for room/wide reconstruction,
                          "2x" for object/telephoto reconstruction,
                          None for backwards-compatible generic lookup.

    Supported JSON keys:
    - `K` or `camera_matrix`: 3x3 matrix
    - `dist_coeffs`, `distortion_coefficients`, or `distortion`: list of coeffs
    - optional `width`, `height`

    After loading, adapts calibration to the target resolution/orientation.
    """
    calib_path = _find_calibration_file(calibration_path, zoom=zoom)

    if calib_path is None:
        print(
            f"[Calibration] No calibration file found "
            f"(zoom={zoom}). Using synthetic fallback for {width}x{height}."
        )
        return make_default_calibration(width, height)

    with open(calib_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    K_data = data.get("K", data.get("camera_matrix"))
    if K_data is None:
        raise KeyError(f"Calibration file {calib_path} is missing `K` or `camera_matrix`.")

    K = np.array(K_data, dtype=np.float64).reshape(3, 3)
    dist_data = data.get(
        "dist_coeffs",
        data.get("distortion_coefficients", data.get("distortion", []))
    )
    dist_coeffs = (
        np.array(dist_data, dtype=np.float64).reshape(-1)
        if dist_data
        else np.zeros((5,), dtype=np.float64)
    )

    # Sanity check: warn if fx and fy differ by more than 5%
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    if abs(fx - fy) / max(fx, fy) > 0.05:
        warnings.warn(
            f"[Calibration] fx={fx:.1f} and fy={fy:.1f} differ by more than 5%. "
            f"This suggests a poor calibration solve. Consider recalibrating. "
            f"RMS in file: {data.get('rms_reprojection_error', 'unknown')}",
            stacklevel=2,
        )

    # Warn if RMS is high
    rms = data.get("rms_reprojection_error")
    if rms is not None and float(rms) > 1.0:
        warnings.warn(
            f"[Calibration] RMS reprojection error is {rms:.4f}px (target: <1.0px). "
            f"Reconstruction accuracy will be reduced. Recalibrate with better images.",
            stacklevel=2,
        )

    calibration = CameraCalibration(
        K=K,
        dist_coeffs=dist_coeffs,
        width=int(data.get("width", width)),
        height=int(data.get("height", height)),
        source=str(calib_path),
    )

    adapted = adapt_calibration_to_image(calibration, width, height)
    print(f"[Calibration] Using: {adapted.source}")
    return adapted


# ---------------------------------------------------------------------------
# Depth Anything V2 — disparity to distance conversion
# ---------------------------------------------------------------------------

def compute_sequence_distance_range(
    depth_paths: list[Path],
    clip_percentiles: tuple[float, float] = (2.0, 98.0),
    max_samples_per_frame: int = 20000,
) -> tuple[float, float]:
    """
    Estimate the sequence-level disparity range for Depth Anything V2 output.
    """
    samples = []
    rng = np.random.default_rng(0)

    for depth_path in depth_paths:
        if not depth_path.exists():
            continue

        raw = np.load(depth_path).astype(np.float32)
        valid = np.isfinite(raw) & (raw > 1e-6)
        if not np.any(valid):
            continue

        disparity = raw[valid]
        if disparity.size > max_samples_per_frame:
            idx = rng.choice(disparity.size, max_samples_per_frame, replace=False)
            disparity = disparity[idx]

        samples.append(disparity)

    if not samples:
        raise FileNotFoundError(
            "No valid depth files were found. "
            "Run midas_depth.py first to generate depth maps."
        )

    all_samples = np.concatenate(samples)
    lo_pct, hi_pct = clip_percentiles
    lo = float(np.percentile(all_samples, lo_pct))
    hi = float(np.percentile(all_samples, hi_pct))

    if hi - lo < 1e-6:
        hi = lo + 1e-3

    print(f"[Depth] Sequence disparity range: lo={lo:.4f}  hi={hi:.4f}  "
          f"(computed from {len(samples)} frames)")
    return lo, hi


def convert_depth_anything_to_distance(
    depth: np.ndarray,
    distance_range: tuple[float, float],
    near: float = 0.3,
    far: float = 8.0,
) -> np.ndarray:
    """Convert Depth Anything V2 relative disparity to pseudo-metric distance."""
    depth = np.asarray(depth, dtype=np.float32)
    distance = np.zeros_like(depth, dtype=np.float32)

    valid = np.isfinite(depth) & (depth > 1e-6)
    if not np.any(valid):
        return distance

    lo, hi = distance_range
    if hi - lo < 1e-6:
        distance[valid] = (near + far) / 2.0
        return distance

    disp_norm = np.clip((depth[valid] - lo) / (hi - lo), 0.0, 1.0)
    distance[valid] = near + (1.0 - disp_norm) * (far - near)
    return distance


def convert_midas_depth_to_distance(
    depth: np.ndarray,
    distance_range: tuple[float, float],
    near: float = 0.3,
    far: float = 8.0,
) -> np.ndarray:
    """Alias for convert_depth_anything_to_distance — kept for backwards compatibility."""
    return convert_depth_anything_to_distance(depth, distance_range, near, far)


def load_midas_distance_map(
    depth_path: Path,
    distance_range: tuple[float, float],
    near: float = 0.3,
    far: float = 8.0,
) -> np.ndarray | None:
    """Load a Depth Anything V2 .npy disparity file and convert to distance."""
    if not depth_path.exists():
        return None

    depth_raw = np.load(depth_path).astype(np.float32)
    return convert_depth_anything_to_distance(
        depth_raw,
        distance_range=distance_range,
        near=near,
        far=far,
    )


def write_point_cloud_ply(
    out_path: Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
) -> None:
    """Write a basic ASCII PLY point cloud without requiring Open3D."""
    points = np.asarray(points, dtype=np.float32)
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float32)
        if colors.shape[0] != points.shape[0]:
            raise ValueError("Point and color counts must match.")
        colors_uint8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    else:
        colors_uint8 = None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors_uint8 is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        if colors_uint8 is None:
            for x, y, z in points:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        else:
            for (x, y, z), (r, g, b) in zip(points, colors_uint8):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")