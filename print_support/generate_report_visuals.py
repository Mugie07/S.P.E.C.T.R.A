from pathlib import Path
import json
import math
import struct

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "print_support"
OUT.mkdir(exist_ok=True)


def font(size=28, bold=False):
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


def fit_image(path, size):
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def make_evidence_strip():
    items = [
        ("Raw phone frame", ROOT / "data/raw_phone/capture_000.jpg"),
        ("Selected keyframe", ROOT / "data/keyframes/frame_000.jpg"),
        ("Depth preview", ROOT / "data/results/depth/frame_000_depth.png"),
        ("Fusion quality preview", ROOT / "data/results/fusion_quality_preview.png"),
    ]
    w, h = 420, 300
    margin = 36
    label_h = 76
    title_h = 76
    canvas = Image.new("RGB", (margin * 2 + w * 4, title_h + h + label_h + margin), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 24), "Real Pipeline Evidence: Capture to Reconstruction Preview", fill=(24, 37, 52), font=font(30, True))

    for i, (label, path) in enumerate(items):
        x = margin + i * w
        y = title_h
        panel = fit_image(path, (w - 18, h))
        canvas.paste(panel, (x + 9, y))
        draw.rectangle((x + 9, y, x + w - 9, y + h), outline=(80, 90, 105), width=2)
        draw.text((x + 14, y + h + 16), f"{i + 1}. {label}", fill=(35, 45, 60), font=font(23, True))
    canvas.save(OUT / "figure_real_pipeline_evidence.png", quality=95)


def load_ply_points(path, max_points=90000):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                return np.empty((0, 3))
            text = line.decode("ascii", errors="ignore").strip()
            header.append(text)
            if text == "end_header":
                break
        vertex_count = 0
        prop_count = 0
        fmt = "ascii"
        in_vertex = False
        vertex_props = []
        for line in header:
            if line.startswith("format"):
                fmt = line.split()[1]
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
                in_vertex = True
                vertex_props = []
            elif line.startswith("element ") and not line.startswith("element vertex"):
                in_vertex = False
            elif in_vertex and line.startswith("property"):
                parts = line.split()
                if len(parts) >= 3:
                    vertex_props.append((parts[1], parts[2]))
        if vertex_count == 0:
            return np.empty((0, 3))
        pts = []
        stride = max(1, math.ceil(vertex_count / max_points))
        if fmt == "ascii":
            for idx in range(vertex_count):
                line = f.readline().decode("ascii", errors="ignore").strip()
                if idx % stride:
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        pts.append((float(parts[0]), float(parts[1]), float(parts[2])))
                    except ValueError:
                        pass
        elif fmt == "binary_little_endian":
            type_map = {
                "float": ("f", 4),
                "float32": ("f", 4),
                "double": ("d", 8),
                "float64": ("d", 8),
                "uchar": ("B", 1),
                "uint8": ("B", 1),
                "char": ("b", 1),
                "int8": ("b", 1),
                "ushort": ("H", 2),
                "uint16": ("H", 2),
                "short": ("h", 2),
                "int16": ("h", 2),
                "uint": ("I", 4),
                "uint32": ("I", 4),
                "int": ("i", 4),
                "int32": ("i", 4),
            }
            fmt_chars = []
            prop_names = []
            for typ, name in vertex_props:
                if typ not in type_map:
                    return np.empty((0, 3))
                fmt_chars.append(type_map[typ][0])
                prop_names.append(name)
            row_fmt = "<" + "".join(fmt_chars)
            row_size = struct.calcsize(row_fmt)
            x_i, y_i, z_i = prop_names.index("x"), prop_names.index("y"), prop_names.index("z")
            for idx in range(vertex_count):
                row = f.read(row_size)
                if len(row) < row_size:
                    break
                if idx % stride:
                    continue
                vals = struct.unpack(row_fmt, row)
                pts.append((float(vals[x_i]), float(vals[y_i]), float(vals[z_i])))
        else:
            return np.empty((0, 3))
    return np.array(pts)


def make_pointcloud_projection():
    files = [
        ("Sparse SfM", ROOT / "data/results/sparse_cloud.ply"),
        ("Dense fused", ROOT / "data/results/dense_fused_cloud.ply"),
        ("Cleaned dense", ROOT / "data/results/dense_fused_cloud_clean.ply"),
        ("Poisson mesh vertices", ROOT / "data/results/mesh_poisson.ply"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), dpi=220)
    fig.suptitle("Real Reconstruction Geometry: Top-View Projection", fontsize=16, fontweight="bold")
    for ax, (title, path) in zip(axes.flat, files):
        pts = load_ply_points(path)
        if pts.size:
            pts = pts[np.isfinite(pts).all(axis=1)]
            ax.scatter(pts[:, 0], pts[:, 1], s=0.35, c=pts[:, 2], cmap="viridis", alpha=0.75, linewidths=0)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{title}\n{len(pts):,} plotted points", fontsize=10)
        else:
            ax.text(0.5, 0.5, "No readable ASCII vertices", ha="center", va="center")
            ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(True)
    fig.text(0.5, 0.02, "Source: S.P.E.C.T.R.A project outputs in data/results", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(OUT / "figure_real_geometry_projection.png", bbox_inches="tight")
    plt.close(fig)


def make_timing_chart():
    stages = [
        ("Keyframe\nExtraction", 4.82),
        ("Prepare\nKeyframes", 3.17),
        ("Depth\nEstimation", 96.44),
        ("Sparse\nSfM", 28.76),
        ("Depth\nFusion", 42.11),
        ("Noise\nCleanup", 8.93),
        ("Surface\nMeshing", 34.58),
        ("Evaluation", 2.05),
        ("Gaussian\nExport", 15.90),
    ]
    labels = [s for s, _ in stages]
    values = [v for _, v in stages]
    total = sum(values)
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=220)
    bars = ax.bar(range(len(values)), values, color="#3f6f8f")
    bars[2].set_color("#b55342")
    ax.set_title("Nine-Stage Pipeline Timing Profile", fontsize=16, fontweight="bold")
    ax.set_ylabel("Mock processing time (seconds)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", color="#d9dee7", linewidth=0.7)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.2f}s", ha="center", va="bottom", fontsize=8)
    ax.text(0.99, 0.93, f"Total: {total:.2f}s", transform=ax.transAxes, ha="right", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "figure_pipeline_timing_profile.png", bbox_inches="tight")
    plt.close(fig)


def make_synthetic_chart():
    report = json.loads((ROOT / "data/results/synthetic/synthetic_report.json").read_text())
    scenes = report["scenes"]
    names = [s["scene"].title() for s in scenes]
    points = [s["point_cloud"]["n_points"] for s in scenes]
    tris = [s["mesh"]["n_triangles"] for s in scenes]
    normal = [s["quality_summary"]["normal_consistency_pct"] for s in scenes]

    x = np.arange(len(names))
    fig, ax1 = plt.subplots(figsize=(10.5, 5.8), dpi=220)
    width = 0.34
    ax1.bar(x - width / 2, np.array(points) / 1000, width, label="Point cloud points (thousands)", color="#406f60")
    ax1.bar(x + width / 2, np.array(tris) / 1000, width, label="Mesh triangles (thousands)", color="#8a5a44")
    ax1.set_ylabel("Count (thousands)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.grid(axis="y", color="#d9dee7", linewidth=0.7)
    ax1.set_axisbelow(True)
    ax2 = ax1.twinx()
    ax2.plot(x, normal, color="#263447", marker="o", linewidth=2.5, label="Normal consistency (%)")
    ax2.set_ylabel("Normal consistency (%)")
    ax2.set_ylim(0, 100)
    fig.suptitle("Synthetic Baseline Reconstruction Comparison", fontsize=16, fontweight="bold")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figure_synthetic_baseline_comparison.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_evidence_strip()
    make_pointcloud_projection()
    make_timing_chart()
    make_synthetic_chart()
