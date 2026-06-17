from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path("E:/3DRealCar_Segment_Dataset")
OUT_DIR = ROOT / "print_support" / "ppt_assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def find_sample() -> tuple[Path, Path, Path]:
    image_dir = DATASET / "images" / "valid"
    ann_dir = DATASET / "annotations" / "valid"
    for image_path in sorted(image_dir.glob("*.png")):
        mask_path = ann_dir / image_path.name
        color_path = ann_dir / image_path.with_name(f"{image_path.stem}_color.png").name
        if mask_path.exists() and color_path.exists():
            return image_path, mask_path, color_path
    raise FileNotFoundError("No matching 3DRealCar valid image and annotation pair found.")


def main() -> None:
    image_path, mask_path, color_path = find_sample()
    image_out = OUT_DIR / "3drealcar_sample_rgb.png"
    mask_out = OUT_DIR / "3drealcar_sample_mask.png"
    panel_out = OUT_DIR / "3drealcar_segmentation_panel.png"

    fit_image(image_path, (1200, 760)).save(image_out)
    fit_image(color_path, (1200, 760)).save(mask_out)

    w, h = 1800, 980
    panel = Image.new("RGB", (w, h), (246, 248, 251))
    draw = ImageDraw.Draw(panel)
    draw.text((70, 55), "3DRealCar Segmentation Dataset", fill=(15, 23, 42), font=font(54, True))
    draw.text(
        (72, 125),
        "Real captured vehicle image with labeled car-part annotation from the valid split",
        fill=(71, 85, 105),
        font=font(28),
    )

    left = fit_image(image_path, (800, 620))
    right = fit_image(color_path, (800, 620))
    panel.paste(left, (70, 230))
    panel.paste(right, (930, 230))
    draw.rounded_rectangle((70, 230, 870, 850), radius=18, outline=(203, 213, 225), width=4)
    draw.rounded_rectangle((930, 230, 1730, 850), radius=18, outline=(203, 213, 225), width=4)
    draw.text((75, 875), "RGB vehicle capture", fill=(15, 23, 42), font=font(30, True))
    draw.text((935, 875), "Semantic car-part mask", fill=(15, 23, 42), font=font(30, True))
    draw.text((70, 928), f"Source file: {image_path.name}", fill=(71, 85, 105), font=font(20))
    panel.save(panel_out)

    print(image_out)
    print(mask_out)
    print(panel_out)


if __name__ == "__main__":
    main()
