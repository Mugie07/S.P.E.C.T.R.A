import cv2
import os
import numpy as np

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def make_rotation(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_REFLECT)


def make_perspective(img, strength=0.15, direction="left"):
    """
    Simulate camera rotating around object by warping corners
    asymmetrically — each corner moves differently.
    """
    h, w = img.shape[:2]
    s = strength

    src = np.float32([[0,0], [w,0], [0,h], [w,h]])

    if direction == "left":
        dst = np.float32([
            [w*s,      h*s*0.5],
            [w,        0      ],
            [w*s,      h      ],
            [w,        h      ]
        ])
    elif direction == "right":
        dst = np.float32([
            [0,        0      ],
            [w*(1-s),  h*s*0.5],
            [0,        h      ],
            [w*(1-s),  h      ]
        ])
    elif direction == "up":
        dst = np.float32([
            [w*s*0.5,  h*s  ],
            [w*(1-s*0.5), h*s],
            [0,        h    ],
            [w,        h    ]
        ])
    elif direction == "down":
        dst = np.float32([
            [0,        0    ],
            [w,        0    ],
            [w*s*0.5,  h*(1-s)],
            [w*(1-s*0.5), h*(1-s)]
        ])
    elif direction == "top_left":
        dst = np.float32([
            [w*s,      h*s  ],
            [w,        0    ],
            [0,        h    ],
            [w,        h    ]
        ])
    elif direction == "top_right":
        dst = np.float32([
            [0,        0    ],
            [w*(1-s),  h*s  ],
            [0,        h    ],
            [w,        h    ]
        ])

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h),
                               borderMode=cv2.BORDER_REFLECT)


def make_zoom(img, scale=0.85):
    """Simulate camera moving closer/further."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), 0, scale)
    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_REFLECT)


def augment_image(img):
    augmented = []

    # 1. Fine rotations — more steps for better baseline
    for angle in [-15, -10, -7, -4, -2, 2, 4, 7, 10, 15]:
        augmented.append(("rot", make_rotation(img, angle)))

    # 2. Real perspective warps — asymmetric corner shifts
    for strength in [0.08, 0.15, 0.22]:
        for direction in ["left", "right", "up", "down",
                          "top_left", "top_right"]:
            augmented.append((
                f"persp_{direction}_{int(strength*100)}",
                make_perspective(img, strength, direction)
            ))

    # 3. Zoom levels
    for scale in [0.80, 0.88, 0.93, 1.08, 1.15]:
        augmented.append(("zoom", make_zoom(img, scale)))

    # 4. Combined: rotation + perspective
    for angle in [-8, 8]:
        for direction in ["left", "right"]:
            rotated = make_rotation(img, angle)
            combined = make_perspective(rotated, 0.10, direction)
            augmented.append((f"combo_{angle}_{direction}", combined))

    return augmented


def find_input_image(input_dir="data/single_image"):
    candidates = []
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No supported image found in {input_dir}. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if len(candidates) == 1:
        return candidates[0]

    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    newest = candidates[0]
    print(
        "Warning: multiple input images found. "
        f"Using newest file: {os.path.basename(newest)}"
    )
    print("Other files still present:")
    for old_path in candidates[1:]:
        print(f"  - {os.path.basename(old_path)}")
    return newest


def clear_output_frames(output_dir="data/keyframes"):
    removed = 0
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if not os.path.isfile(path):
            continue

        stem, ext = os.path.splitext(name)
        if stem.startswith("frame_") and ext.lower() in SUPPORTED_EXTENSIONS:
            os.remove(path)
            removed += 1

    return removed


def main():
    input_path = find_input_image("data/single_image")
    output_dir = "data/keyframes"          # ← writes directly to keyframes

    os.makedirs(output_dir, exist_ok=True)
    removed = clear_output_frames(output_dir)

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {input_path}")

    print(f"Using input: {input_path}")
    if removed:
        print(f"Cleared {removed} old frame files from {output_dir}")
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    # Save original as frame_000
    cv2.imwrite(os.path.join(output_dir, "frame_000.jpg"), img)
    print("Saved original: frame_000.jpg")

    augmented = augment_image(img)

    for i, (tag, aug_img) in enumerate(augmented):
        filename = f"frame_{i+1:03d}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), aug_img)

    total = len(augmented) + 1
    print(f"Saved {total} images to {output_dir}/")
    print(f"  - 1  original")
    print(f"  - 10 rotation variants")
    print(f"  - 18 perspective variants")
    print(f"  - 5  zoom variants")
    print(f"  - 4  combined variants")
    print(f"\nNext: run sparse_recon_live.py")


if __name__ == "__main__":
    main()
