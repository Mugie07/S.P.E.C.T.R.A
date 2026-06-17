from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def load_depth_anything_v2(device: torch.device):
    try:
        image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Depth Anything V2 from Hugging Face. "
            "Make sure this environment has internet access at least once, "
            "or that the model is already cached locally."
        ) from exc
    model.to(device)
    model.eval()
    return model, image_processor


def normalize_to_uint8(depth: np.ndarray) -> np.ndarray:
    """For visualization only - does NOT affect saved .npy values."""
    d = depth.astype(np.float32)
    d_min, d_max = np.min(d), np.max(d)
    if d_max - d_min < 1e-8:
        return np.zeros_like(d, dtype=np.uint8)
    return ((d - d_min) / (d_max - d_min) * 255.0).clip(0, 255).astype(np.uint8)


def clear_old_depth_outputs(output_dir: Path) -> int:
    removed = 0
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        if path.name.endswith("_depth.npy") or path.name.endswith("_depth.png"):
            path.unlink()
            removed += 1
    return removed


def main() -> None:
    input_dir = Path("data/keyframes")
    output_dir = Path("data/results/depth")
    output_dir.mkdir(parents=True, exist_ok=True)
    removed = clear_old_depth_outputs(output_dir)

    image_paths = sorted(
        list(input_dir.glob("*.jpg")) +
        list(input_dir.glob("*.png")) +
        list(input_dir.glob("*.jpeg"))
    )

    print(f"DEBUG: Found {len(image_paths)} images in {input_dir}")

    if not image_paths:
        raise FileNotFoundError(f"No images in {input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if removed:
        print(f"Cleared {removed} old depth files from: {output_dir}")

    print(f"Loading Depth Anything V2 model: {MODEL_ID}")
    model, image_processor = load_depth_anything_v2(device)

    print(f"Processing {len(image_paths)} images...")

    for idx, img_path in enumerate(image_paths):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"  Skipping unreadable: {img_path.name}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inputs = image_processor(images=img_rgb, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
            pred = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = pred.cpu().numpy().astype(np.float32)

        np.save(output_dir / f"{img_path.stem}_depth.npy", depth)

        vis = cv2.applyColorMap(normalize_to_uint8(depth), cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(output_dir / f"{img_path.stem}_depth.png"), vis)

        print(
            f"  [{idx + 1}/{len(image_paths)}] {img_path.name} "
            f"| disparity min={depth.min():.4f} max={depth.max():.4f} "
            f"mean={depth.mean():.4f}"
        )

    print(f"\nDone. Depth maps saved to: {output_dir}")


if __name__ == "__main__":
    main()
