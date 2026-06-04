"""Local Flask web service for AnalogVision."""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

try:
    from moviepy.editor import AudioFileClip, VideoFileClip
except ImportError:  # MoviePy 2.x
    from moviepy import AudioFileClip, VideoFileClip

from audio_effects import degrade_audio_clip
from utils import (
    CancelledError,
    DEFAULT_INTENSITY,
    cleanup_temp_dir,
    ensure_input_video,
    ensure_output_path,
    find_ffmpeg_note,
    make_temp_dir,
    normalize_style,
    progress,
    safe_stem,
)
from video_effects import convert_video


ROOT = Path(__file__).resolve().parent
STORAGE = ROOT / "web_storage"
UPLOADS = STORAGE / "uploads"
OUTPUTS = STORAGE / "outputs"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
WEB_MAX_OUTPUT_WIDTH = 960
GIF_WIDTH = 640
GIF_FPS = 8
GIF_SECONDS = 4

app = Flask(__name__)
for directory in (UPLOADS, OUTPUTS):
    directory.mkdir(parents=True, exist_ok=True)

jobs: dict[str, dict] = {}


def run_conversion(
    input_path: str,
    output_path: str,
    style: str,
    convert_4_3: bool = True,
    degrade_audio: bool = True,
    max_output_width: int | None = None,
    progress_callback=None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Path:
    """Run video conversion for the web app."""

    source = ensure_input_video(input_path)
    output = ensure_output_path(output_path)
    style = normalize_style(style)
    intensity = DEFAULT_INTENSITY

    temp_dir = make_temp_dir()
    clips_to_close = []
    try:
        progress(progress_callback, 0.02, "Preparing temporary workspace")
        processed_video = temp_dir / f"{safe_stem(source)}_styled.mp4"

        meta = convert_video(
            source,
            processed_video,
            style=style,
            intensity=intensity,
            convert_4_3=convert_4_3,
            max_output_width=max_output_width,
            progress_callback=lambda p, m: progress(progress_callback, 0.05 + p * 0.62, m),
            cancel_check=cancel_check,
        )

        main_clip = VideoFileClip(str(processed_video))
        clips_to_close.append(main_clip)
        final_video = main_clip

        source_audio_clip = None
        try:
            source_audio_clip = AudioFileClip(str(source))
            clips_to_close.append(source_audio_clip)
        except Exception:
            source_audio_clip = None

        final_audio = None
        if source_audio_clip is not None:
            if degrade_audio:
                progress(progress_callback, 0.80, "Degrading original audio")
                final_audio = degrade_audio_clip(source_audio_clip, style, intensity)
            else:
                final_audio = source_audio_clip
            if final_audio is not source_audio_clip and final_audio is not None:
                clips_to_close.append(final_audio)

        if final_audio is not None:
            final_video = _set_audio(final_video, final_audio)

        progress(progress_callback, 0.90, "Encoding final MP4")
        final_video.write_videofile(
            str(output),
            codec="libx264",
            audio_codec="aac",
            fps=meta["fps"],
            preset="medium",
            threads=2,
            logger=None,
        )
        progress(progress_callback, 1.0, f"Done: {output}")
        return output
    except CancelledError:
        progress(progress_callback, 0.0, "Cancelled")
        raise
    except Exception as exc:
        raise RuntimeError(f"{exc}\n\n{find_ffmpeg_note()}") from exc
    finally:
        for clip in reversed(clips_to_close):
            try:
                clip.close()
            except Exception:
                pass
        cleanup_temp_dir(temp_dir)


def _set_audio(video_clip, audio_clip):
    if hasattr(video_clip, "set_audio"):
        return video_clip.set_audio(audio_clip)
    return video_clip.with_audio(audio_clip)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/convert")
def convert():
    upload = request.files.get("video")
    if not upload or not upload.filename:
        return jsonify({"error": "Choose a video file."}), 400

    suffix = Path(upload.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported video format."}), 400

    job_id = uuid.uuid4().hex
    safe_name = secure_filename(upload.filename) or f"source{suffix}"
    input_name = f"{job_id}_{safe_name}"
    output_name = f"{Path(safe_name).stem}_analogvision_{job_id[:8]}.mp4"
    gif_name = f"{Path(safe_name).stem}_before_after_{job_id[:8]}.gif"
    input_path = UPLOADS / input_name
    output_path = OUTPUTS / output_name
    gif_path = OUTPUTS / gif_name
    upload.save(input_path)

    style = request.form.get("style", "VHS 80s")
    convert_43 = request.form.get("convert43") == "true"
    degrade_audio = request.form.get("degradeAudio") == "true"

    jobs[job_id] = {
        "id": job_id,
        "state": "queued",
        "progress": 0.0,
        "message": "Queued",
        "eta_seconds": None,
        "elapsed_seconds": 0,
        "input_url": f"/media/uploads/{input_name}",
        "output_url": None,
        "download_url": None,
        "gif_url": None,
        "gif_download_url": None,
    }

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, input_path, output_path, gif_path, style, convert_43, degrade_audio),
        daemon=True,
    )
    thread.start()
    return jsonify({"job_id": job_id})


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404
    return jsonify(job)


@app.get("/media/<kind>/<path:filename>")
def media(kind: str, filename: str):
    if kind == "uploads":
        return send_from_directory(UPLOADS, filename)
    if kind == "outputs":
        return send_from_directory(OUTPUTS, filename)
    return jsonify({"error": "Not found."}), 404


def _run_job(
    job_id: str,
    input_path: Path,
    output_path: Path,
    gif_path: Path,
    style: str,
    convert_43: bool,
    degrade_audio: bool,
) -> None:
    started_at = time.monotonic()

    def on_progress(value: float, message: str) -> None:
        elapsed = max(0.0, time.monotonic() - started_at)
        safe_value = max(0.0, min(1.0, value))
        eta = None
        if safe_value >= 0.04:
            eta = max(0, int((elapsed / safe_value) - elapsed))
        jobs[job_id].update(
            {
                "state": "running",
                "progress": round(safe_value, 4),
                "message": message,
                "eta_seconds": eta,
                "elapsed_seconds": int(elapsed),
            }
        )

    try:
        run_conversion(
            input_path=str(input_path),
            output_path=str(output_path),
            style=style,
            convert_4_3=convert_43,
            degrade_audio=degrade_audio,
            max_output_width=WEB_MAX_OUTPUT_WIDTH,
            progress_callback=on_progress,
        )
        on_progress(0.97, "Creating before/after GIF")
        _make_comparison_gif(input_path, output_path, gif_path)
        jobs[job_id].update(
            {
                "state": "complete",
                "progress": 1.0,
                "message": "Done",
                "eta_seconds": 0,
                "elapsed_seconds": int(time.monotonic() - started_at),
                "output_url": f"/media/outputs/{output_path.name}",
                "download_url": f"/media/outputs/{output_path.name}",
                "gif_url": f"/media/outputs/{gif_path.name}",
                "gif_download_url": f"/media/outputs/{gif_path.name}",
            }
        )
    except Exception as exc:
        jobs[job_id].update({"state": "error", "message": str(exc), "progress": 0.0})


def _make_comparison_gif(before_path: Path, after_path: Path, gif_path: Path) -> None:
    before = cv2.VideoCapture(str(before_path))
    after = cv2.VideoCapture(str(after_path))
    try:
        if not before.isOpened() or not after.isOpened():
            raise RuntimeError("Could not open videos for GIF export.")

        before_fps = before.get(cv2.CAP_PROP_FPS) or 30.0
        after_fps = after.get(cv2.CAP_PROP_FPS) or before_fps
        before_frames = int(before.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        after_frames = int(after.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_total = min(GIF_SECONDS * GIF_FPS, before_frames, after_frames)
        if frame_total <= 0:
            raise RuntimeError("No frames available for GIF export.")

        width = GIF_WIDTH
        height = int(width * 9 / 16)
        frames = []
        for i in range(frame_total):
            before.set(cv2.CAP_PROP_POS_FRAMES, int(i * before_fps / GIF_FPS))
            after.set(cv2.CAP_PROP_POS_FRAMES, int(i * after_fps / GIF_FPS))
            ok_before, before_frame = before.read()
            ok_after, after_frame = after.read()
            if not ok_before or not ok_after:
                break
            frames.append(_comparison_frame(before_frame, after_frame, width, height))

        if not frames:
            raise RuntimeError("No frames available for GIF export.")

        duration_ms = int(1000 / GIF_FPS)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=duration_ms,
            loop=0,
        )
    finally:
        before.release()
        after.release()


def _comparison_frame(before_frame, after_frame, width: int, height: int) -> Image.Image:
    before_rgb = cv2.cvtColor(_cover_frame(before_frame, width, height), cv2.COLOR_BGR2RGB)
    after_rgb = cv2.cvtColor(_cover_frame(after_frame, width, height), cv2.COLOR_BGR2RGB)
    split = width // 2
    combined = before_rgb.copy()
    combined[:, split:] = after_rgb[:, split:]
    combined[:, max(0, split - 1) : split + 1] = [236, 239, 222]
    return Image.fromarray(combined)


def _cover_frame(frame, width: int, height: int):
    h, w = frame.shape[:2]
    scale = max(width / w, height / h)
    resized_w = max(width, int(round(w * scale)))
    resized_h = max(height, int(round(h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    x0 = (resized_w - width) // 2
    y0 = (resized_h - height) // 2
    return resized[y0 : y0 + height, x0 : x0 + width]


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
