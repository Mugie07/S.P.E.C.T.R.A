from __future__ import annotations

from pathlib import Path

import open3d as o3d


def find_single_dense_cloud(results_dir: Path, keyframes_dir: Path) -> Path:
    image_stems = [
        path.stem
        for path in sorted(
            list(keyframes_dir.glob("*.jpg")) +
            list(keyframes_dir.glob("*.png")) +
            list(keyframes_dir.glob("*.jpeg"))
        )
    ]
    for stem in image_stems:
        candidate = results_dir / f"{stem}_dense_cloud.ply"
        if candidate.exists():
            return candidate

    preferred = results_dir / "frame_000_dense_cloud.ply"
    candidates = sorted(results_dir.glob("*_dense_cloud.ply"))
    candidates = [path for path in candidates if path.name != "dense_fused_cloud.ply"]
    if candidates:
        return candidates[0]
    return preferred


def count_points(ply_path: Path) -> int:
    if not ply_path.exists():
        return -1
    pcd = o3d.io.read_point_cloud(str(ply_path))
    return len(pcd.points)


def status_text(count: int) -> str:
    if count < 0:
        return "Missing"
    if count == 0:
        return "Empty"
    return "Available"


def main() -> None:
    results_dir = Path("data/results")
    keyframes_dir = Path("data/keyframes")
    sparse_path = results_dir / "sparse_cloud.ply"
    single_dense_path = find_single_dense_cloud(results_dir, keyframes_dir)
    fused_dense_path = results_dir / "dense_fused_cloud.ply"

    sparse_count = count_points(sparse_path)
    single_dense_count = count_points(single_dense_path)
    fused_dense_count = count_points(fused_dense_path)

    print("\n=== Reconstruction Evaluation Report ===\n")

    print(f"Sparse cloud file:        {sparse_path}")
    print(f"Status:                   {status_text(sparse_count)}")
    print(f"Point count:              {sparse_count if sparse_count >= 0 else 'N/A'}\n")

    print(f"Single dense cloud file:  {single_dense_path}")
    print(f"Status:                   {status_text(single_dense_count)}")
    print(f"Point count:              {single_dense_count if single_dense_count >= 0 else 'N/A'}\n")

    print(f"Fused dense cloud file:   {fused_dense_path}")
    print(f"Status:                   {status_text(fused_dense_count)}")
    print(f"Point count:              {fused_dense_count if fused_dense_count >= 0 else 'N/A'}\n")

    print("=== Comparative Summary ===\n")

    if sparse_count > 0 and single_dense_count > 0:
        ratio = single_dense_count / sparse_count
        print(f"Single-image dense cloud has {ratio:.2f}x more points than sparse cloud.")

    if sparse_count > 0 and fused_dense_count > 0:
        ratio = fused_dense_count / sparse_count
        print(f"Fused dense cloud has {ratio:.2f}x more points than sparse cloud.")

    if single_dense_count > 0 and fused_dense_count > 0:
        ratio = fused_dense_count / single_dense_count
        print(f"Fused dense cloud has {ratio:.2f}x more points than single-image dense cloud.")

    print("\nInterpretation:")
    print("- Sparse reconstruction captures only matched feature points.")
    print("- Single-image dense reconstruction uses AI depth estimation for one view.")
    print("- Fused dense reconstruction combines multiple AI-derived depth clouds for richer geometry.")

    print("\nDone.")


if __name__ == "__main__":
    main()
