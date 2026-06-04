"""Audio degradation effects."""

from __future__ import annotations

import numpy as np

try:
    from moviepy.audio.AudioClip import AudioArrayClip
except ImportError:  # MoviePy package layouts vary slightly.
    from moviepy import AudioArrayClip

from scipy.signal import butter, lfilter

from utils import DEFAULT_INTENSITY, PROFILES


def degrade_audio_clip(source_clip, style: str, intensity: str = DEFAULT_INTENSITY, sample_rate: int = 44100):
    """Return a MoviePy AudioArrayClip with muffled, noisy analog-style audio."""

    if source_clip is None:
        return None

    profile = PROFILES[intensity]
    duration = float(source_clip.duration or 0)
    if duration <= 0:
        return None

    audio = source_clip.to_soundarray(fps=sample_rate)
    if audio.ndim == 1:
        audio = audio[:, None]
    audio = audio.astype(np.float32)

    # Narrow stereo toward mono, like consumer tape playback.
    mono = np.mean(audio, axis=1, keepdims=True)
    narrow = 0.70 if style == "VHS 80s" else 0.45
    audio = mono * narrow + audio * (1.0 - narrow)

    cutoff = 2600 if style == "VHS 80s" else 3600
    cutoff = max(1200, cutoff - int(profile.scale * 650))
    audio = _lowpass(audio, cutoff, sample_rate)

    # Gentle saturation: a cheap analog preamp and tape do not clip digitally.
    drive = 1.4 + profile.scale * 0.45
    audio = np.tanh(audio * drive) / np.tanh(drive)

    t = np.arange(len(audio), dtype=np.float32) / sample_rate
    wow = 1.0 + 0.035 * profile.scale * np.sin(2 * np.pi * 0.55 * t)
    flutter = 1.0 + 0.010 * profile.scale * np.sin(2 * np.pi * 7.2 * t)
    volume = (wow * flutter)[:, None]
    audio *= volume

    rng = np.random.default_rng(1984)
    hiss = rng.normal(0, 0.008 + profile.scale * 0.006, audio.shape).astype(np.float32)
    hum = (np.sin(2 * np.pi * 60 * t) * (0.006 + profile.scale * 0.002))[:, None]
    audio = audio + hiss + hum

    # A crude "lower sample rate feel": quantize amplitude and lightly decimate/rebuild.
    steps = 64
    audio = np.round(audio * steps) / steps
    if profile.scale >= 1.0:
        factor = 2
        short = audio[::factor]
        x_old = np.linspace(0, 1, len(short))
        x_new = np.linspace(0, 1, len(audio))
        audio = np.vstack([np.interp(x_new, x_old, short[:, c]) for c in range(short.shape[1])]).T

    audio = np.clip(audio, -0.98, 0.98)
    return AudioArrayClip(audio, fps=sample_rate)


def _lowpass(audio: np.ndarray, cutoff_hz: int, sample_rate: int) -> np.ndarray:
    nyquist = sample_rate / 2
    b, a = butter(3, min(0.98, cutoff_hz / nyquist), btype="low")
    return lfilter(b, a, audio, axis=0).astype(np.float32)
