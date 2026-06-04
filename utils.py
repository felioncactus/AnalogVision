"""Shared helpers for the VHS / B&W converter."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


VALID_STYLES = ("VHS 80s", "artistic B&W")
DEFAULT_INTENSITY = "realistic"


@dataclass(frozen=True)
class EffectProfile:
    """Numbers that control the strength of synthetic analog defects."""

    scale: float
    grain: float
    blur: float
    chroma_shift: int
    jitter: int
    wobble: int
    dropout: float
    ghost: float
    scratch: float
    flicker: float
    vignette: float
    tracking: float
    halation: float


PROFILES = {
    # Single full-strength profile: worn enough to read as real tape playback,
    # bounded enough to avoid turning into abstract glitch art.
    DEFAULT_INTENSITY: EffectProfile(1.12, 0.052, 1.08, 4, 6, 4, 0.074, 0.150, 0.018, 0.062, 0.30, 0.076, 0.22),
}


ProgressCallback = Optional[Callable[[float, str], None]]


class CancelledError(RuntimeError):
    """Raised when the GUI asks the processing loop to stop."""


def normalize_style(style: str) -> str:
    text = (style or "").strip().lower()
    aliases = {
        "vhs": "VHS 80s",
        "vhs 80s": "VHS 80s",
        "80s": "VHS 80s",
        "black and white": "artistic B&W",
        "black and white art": "artistic B&W",
        "bw": "artistic B&W",
        "b&w": "artistic B&W",
        "art bw": "artistic B&W",
        "art b&w": "artistic B&W",
        "artistic bw": "artistic B&W",
        "artistic b&w": "artistic B&W",
    }
    if text not in aliases:
        raise ValueError(f"Unknown style '{style}'. Choose one of: {', '.join(VALID_STYLES)}")
    return aliases[text]


def ensure_input_video(path: str) -> Path:
    video = Path(path).expanduser()
    if not video.exists() or not video.is_file():
        raise FileNotFoundError(f"Input video not found: {video}")
    return video


def ensure_output_path(path: str) -> Path:
    output = Path(path).expanduser()
    if not output.suffix:
        output = output.with_suffix(".mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def make_temp_dir(prefix: str = "analogvision_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def cleanup_temp_dir(path: Path) -> None:
    if path and path.exists():
        shutil.rmtree(path, ignore_errors=True)


def progress(callback: ProgressCallback, value: float, message: str) -> None:
    value = max(0.0, min(1.0, float(value)))
    if callback:
        callback(value, message)
    else:
        print(f"[{value * 100:6.2f}%] {message}", flush=True)


def check_cancel(cancel_check: Optional[Callable[[], bool]]) -> None:
    if cancel_check and cancel_check():
        raise CancelledError("Conversion cancelled by user.")


def find_ffmpeg_note() -> str:
    return (
        "MoviePy needs ffmpeg. Most MoviePy installs download/use imageio-ffmpeg "
        "automatically; if export fails, install ffmpeg and make sure it is on PATH."
    )


def safe_stem(path: os.PathLike[str] | str) -> str:
    return Path(path).stem.replace(" ", "_")
