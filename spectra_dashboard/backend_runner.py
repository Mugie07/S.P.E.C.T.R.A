from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
DEPTH_DIR = RESULTS_DIR / "depth"
SELECTED_DIR = DATA_DIR / "selected"
KEYFRAMES_DIR = DATA_DIR / "keyframes"
GAUSSIAN_SCENE_DIR = DATA_DIR / "gaussian_splatting_scene"
LOG_DIR = DATA_DIR / "logs"
LIVE_STATUS = LOG_DIR / "live_status.json"
LIVE_LOG = LOG_DIR / "live_pipeline.log"

LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Stage:
    key: str
    name: str
    command: tuple[str, ...]
    output_keys: tuple[str, ...]


OUTPUTS = {
    "selected": SELECTED_DIR,
    "keyframes": KEYFRAMES_DIR,
    "depth": DEPTH_DIR,
    "sparse": RESULTS_DIR / "sparse_cloud.ply",
    "poses": RESULTS_DIR / "camera_poses.json",
    "dense": RESULTS_DIR / "dense_fused_cloud_pose_aligned.ply",
    "legacy_dense": RESULTS_DIR / "dense_fused_cloud.ply",
    "clean": RESULTS_DIR / "dense_fused_cloud_clean.ply",
    "mesh": RESULTS_DIR / "mesh_poisson.ply",
    "mesh_obj": RESULTS_DIR / "mesh_poisson.obj",
    "gaussian": GAUSSIAN_SCENE_DIR,
}

STAGES = [
    Stage("selection", "Smart Frame Selection", ("python", "src/utils/smart_frame_selector.py", "--mode", "images"), ("selected",)),
    Stage("prepare", "Prepare Keyframes", ("python", "src/capture/prepare_keyframes_from_folder.py"), ("keyframes",)),
    Stage("depth", "Depth Estimation", ("python", "src/depth/midas_depth.py"), ("depth",)),
    Stage("sfm", "Sparse SfM Reconstruction", ("python", "src/sfm/sparse_recon_live.py", "--no_view"), ("sparse", "poses")),
    Stage("fusion", "Pose-Aligned Depth Fusion", ("python", "src/depth/fuse_depth_clouds_pose_aligned.py", "--zoom", "2x", "--profile", "object"), ("dense",)),
    Stage("cleanup", "Floating Cluster Cleanup", ("python", "src/utils/cleanup.py"), ("clean",)),
    Stage("mesh", "Surface Meshing", ("python", "src/mesh/mesh_surface.py", "--no_view"), ("mesh", "mesh_obj")),
    Stage("legacy_fusion", "Final Dense Cloud Export", ("python", "src/depth/fuse_depth_clouds.py", "--no_view"), ("legacy_dense",)),
    Stage("evaluation", "Evaluation Report", ("python", "src/utils/evaluate_reconstruction.py"), tuple()),
    Stage("gaussian", "Gaussian Export", ("python", "src/export/gaussian_export.py"), ("gaussian",)),
]

SKIPPED_STAGE_KEYS: set[str] = set()
STAGE_BY_KEY = {stage.key: stage for stage in STAGES}


def has_output(key: str) -> bool:
    path = OUTPUTS[key]
    if key == "depth":
        return any(path.glob("*_depth.npy"))
    if path.is_dir():
        return path.exists() and any(path.rglob("*"))
    return path.exists() and path.stat().st_size > 0


def stage_done(stage: Stage, quality_mode: str = "fast") -> bool:
    if stage.key in skipped_stage_keys(quality_mode):
        return True
    if not stage.output_keys:
        return False
    return all(has_output(key) for key in stage.output_keys)


def skipped_stage_keys(quality_mode: str) -> set[str]:
    return {"selection"} if quality_mode == "fast" else set()


def active_stages(stages: list[Stage], quality_mode: str) -> list[Stage]:
    skipped = skipped_stage_keys(quality_mode)
    return [stage for stage in stages if stage.key not in skipped]


def stage_command(stage: Stage, scene_profile: str, zoom_profile: str) -> tuple[str, ...]:
    if stage.key == "selection":
        max_frames = "60" if scene_profile == "lobby" else "24"
        return ("python", "src/utils/smart_frame_selector.py", "--mode", "images", "--max_frames", max_frames)
    if stage.key == "prepare":
        max_images = "60" if scene_profile == "lobby" else "25"
        return ("python", "src/capture/prepare_keyframes_from_folder.py", "--max_images", max_images)
    if stage.key != "fusion":
        return stage.command

    command = [
        "python",
        "src/depth/fuse_depth_clouds_pose_aligned.py",
        "--zoom",
        zoom_profile,
        "--profile",
        "room" if scene_profile == "lobby" else "object",
    ]
    return tuple(command)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def append_log(message: str) -> None:
    with LIVE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def write_status(status: dict) -> None:
    status["updated_at"] = now()
    payload = json.dumps(status, indent=2)
    last_error: Exception | None = None
    for attempt in range(8):
        tmp = LIVE_STATUS.with_name(f"{LIVE_STATUS.stem}.{os.getpid()}.{attempt}.tmp")
        try:
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(LIVE_STATUS)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.15)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
    if last_error is not None:
        append_log(f"[{now()}] WARNING: could not update live status after retries: {last_error}")


def base_status(stage_keys: list[str], mode: str, quality_mode: str, scene_profile: str, zoom_profile: str) -> dict:
    selected = set(stage_keys)
    skipped = skipped_stage_keys(quality_mode)
    return {
        "runner_pid": os.getpid(),
        "mode": mode,
        "quality_mode": quality_mode,
        "scene_profile": scene_profile,
        "zoom_profile": zoom_profile,
        "running": True,
        "started_at": now(),
        "finished_at": None,
        "current_stage": None,
        "last_message": "Backend runner started.",
        "stage_order": [stage.key for stage in STAGES],
        "selected_stages": stage_keys,
        "stages": {
            stage.key: {
                "name": stage.name,
                "status": "skipped" if stage.key in skipped else ("pending" if stage.key in selected else "skipped"),
                "command": " ".join(stage_command(stage, scene_profile, zoom_profile)),
                "started_at": None,
                "finished_at": None,
                "elapsed_s": None,
                "returncode": None,
            }
            for stage in STAGES
        },
    }


def run_stage(stage: Stage, status: dict) -> bool:
    stage_status = status["stages"][stage.key]
    stage_status["status"] = "running"
    stage_status["started_at"] = now()
    status["current_stage"] = stage.key
    status["last_message"] = f"Running {stage.name}"
    write_status(status)

    append_log("")
    append_log(f"[{now()}] START {stage.name}")
    command_tuple = stage_command(
        stage,
        str(status.get("scene_profile", "object")),
        str(status.get("zoom_profile", "2x")),
    )
    append_log(f"[command] {' '.join(command_tuple)}")

    command = list(command_tuple)
    if command[0] == "python":
        command[0] = sys.executable

    env = {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT),
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "SPECTRA_FRAME_SOURCE": "selected" if status.get("quality_mode") == "quality" else "raw_phone",
    }
    started = time.time()
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        append_log(line.rstrip())
        status["last_message"] = line.rstrip()[:500] or f"Running {stage.name}"
        write_status(status)

    returncode = process.wait()
    elapsed = round(time.time() - started, 2)

    stage_status["finished_at"] = now()
    stage_status["elapsed_s"] = elapsed
    stage_status["returncode"] = returncode
    if returncode == 0:
        stage_status["status"] = "done"
        status["last_message"] = f"{stage.name} completed in {elapsed}s."
        append_log(f"[{now()}] DONE {stage.name} ({elapsed}s)")
        write_status(status)
        return True

    stage_status["status"] = "failed"
    status["last_message"] = f"{stage.name} failed with exit code {returncode}."
    append_log(f"[{now()}] FAILED {stage.name} ({elapsed}s, exit {returncode})")
    write_status(status)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S.P.E.C.T.R.A backend stages and publish live status.")
    parser.add_argument("--mode", choices=("all", "remaining", "selected"), default="selected")
    parser.add_argument("--quality_mode", choices=("fast", "quality"), default="fast")
    parser.add_argument("--scene_profile", choices=("object", "lobby"), default="object")
    parser.add_argument("--zoom_profile", choices=("1x", "2x"), default="2x")
    parser.add_argument("--stages", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "all":
        stages = active_stages(STAGES, args.quality_mode)
    elif args.mode == "remaining":
        stages = active_stages([stage for stage in STAGES if not stage_done(stage, args.quality_mode)], args.quality_mode)
    else:
        stages = active_stages([STAGE_BY_KEY[key] for key in args.stages if key in STAGE_BY_KEY], args.quality_mode)

    stage_keys = [stage.key for stage in stages]
    LIVE_LOG.write_text("", encoding="utf-8")
    status = base_status(stage_keys, args.mode, args.quality_mode, args.scene_profile, args.zoom_profile)
    write_status(status)
    append_log(f"[{now()}] Backend runner PID {os.getpid()} started in {args.mode} mode.")
    append_log(f"[{now()}] Python executable: {sys.executable}")
    try:
        import torch

        cuda_message = f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}"
        if torch.cuda.is_available():
            cuda_message += f" device={torch.cuda.get_device_name(0)}"
        append_log(f"[{now()}] {cuda_message}")
    except Exception as exc:
        append_log(f"[{now()}] Torch/CUDA check unavailable: {exc}")
    if args.quality_mode == "quality":
        append_log(f"[{now()}] Smart Frame Selection enabled; Prepare Keyframes reads data/selected.")
    else:
        append_log(f"[{now()}] SKIP Smart Frame Selection - fast mode; Prepare Keyframes reads data/raw_phone directly.")
    append_log(f"[{now()}] Scene profile: {args.scene_profile}; zoom calibration: {args.zoom_profile}.")

    ok = True
    try:
        for stage in stages:
            ok = run_stage(stage, status)
            if not ok:
                break
    except Exception:
        ok = False
        append_log(f"[{now()}] Backend runner crashed:")
        append_log(traceback.format_exc())
        current_stage = status.get("current_stage")
        if current_stage and current_stage in status.get("stages", {}):
            status["stages"][current_stage]["status"] = "failed"
            status["stages"][current_stage]["finished_at"] = now()
        status["last_message"] = "Backend runner crashed. Check live backend log."

    status["running"] = False
    status["current_stage"] = None
    status["finished_at"] = now()
    if ok:
        status["last_message"] = "Backend pipeline complete."
        append_log(f"[{now()}] Backend pipeline complete.")
    else:
        append_log(f"[{now()}] Backend pipeline stopped after failure.")
    write_status(status)


if __name__ == "__main__":
    main()
