"""Frame-by-frame classical video effects for analog VHS and B&W looks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from utils import DEFAULT_INTENSITY, PROFILES, check_cancel, progress


@dataclass
class VHSState:
    """State that persists across frames during one VHS conversion.

    Real tape errors are temporally continuous: tracking drifts, chroma phase
    wanders, and dropouts often arrive in short bursts. Keeping this state
    avoids feeding final display damage back into the next frame.
    """

    previous_clean: Optional[np.ndarray] = None
    previous_signal: Optional[np.ndarray] = None
    chroma_phase: float = 0.0
    tracking_phase: float = 0.0
    head_switch_phase: float = 0.0
    color_gain_phase: float = 0.0
    dropout_burst: int = 0
    tracking_burst: int = 0


def convert_video(
    input_path: str | Path,
    output_video_path: str | Path,
    style: str,
    intensity: str = DEFAULT_INTENSITY,
    convert_4_3: bool = True,
    max_output_width: int | None = None,
    progress_callback=None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> dict:
    """Process a video without audio and write a temporary styled video file."""

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    # Preserve FPS so the original video duration stays intact. The visual
    # effects still simulate 24 fps film cadence and 29.97-ish VHS timing.
    target_fps = fps

    out_w, out_h = (width, height)
    if convert_4_3:
        out_w, out_h = _four_by_three_size(width, height)
    out_w, out_h = _cap_output_size(out_w, out_h, max_output_width)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, target_fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_video_path}")

    profile = PROFILES[intensity]
    rng = np.random.default_rng(1977)
    previous = None
    vhs_state = VHSState() if style == "VHS 80s" else None
    frame_index = 0

    try:
        while True:
            check_cancel(cancel_check)
            ok, frame = cap.read()
            if not ok:
                break

            frame = _fit_to_4_3_smart(frame, out_w, out_h) if convert_4_3 else cv2.resize(frame, (out_w, out_h))
            styled = apply_analog_effects(
                frame,
                frame_index=frame_index,
                fps=target_fps,
                style=style,
                profile=profile,
                previous_frame=previous,
                rng=rng,
                vhs_state=vhs_state,
            )
            writer.write(styled)
            previous = styled.copy()
            frame_index += 1

            if frame_index % 10 == 0 or frame_index == total:
                progress(progress_callback, frame_index / total, f"Processing frame {frame_index}/{total}")
    except Exception:
        raise
    finally:
        cap.release()
        writer.release()

    return {
        "fps": target_fps,
        "width": out_w,
        "height": out_h,
        "frames": frame_index,
        "duration": frame_index / target_fps if target_fps else 0,
    }


def apply_analog_effects(
    frame: np.ndarray,
    frame_index: int,
    fps: float,
    style: str,
    profile,
    previous_frame: Optional[np.ndarray],
    rng: np.random.Generator,
    vhs_state: Optional[VHSState] = None,
) -> np.ndarray:
    """Apply a physics-inspired stack of old analog video and film artifacts."""

    img = frame.astype(np.float32) / 255.0

    if style == "VHS 80s":
        if vhs_state is None:
            vhs_state = VHSState()
        img = _vhs_signal_pipeline(img, frame_index, fps, profile, vhs_state, rng)
    elif style == "artistic B&W":
        img = _artistic_black_and_white(img, frame_index, profile, rng)
    else:
        raise ValueError("Style must be 'VHS 80s' or 'artistic B&W'.")

    if style == "artistic B&W":
        img = _film_grain(img, profile, rng, amount=0.42)
        img = _vignette(img, profile)
    img = np.clip(img, 0.0, 1.0)

    out = (img * 255.0).astype(np.uint8)
    return out


def _four_by_three_size(width: int, height: int) -> tuple[int, int]:
    target_w = width
    target_h = int(round(target_w * 3 / 4))
    if target_h > height:
        target_h = height
        target_w = int(round(target_h * 4 / 3))
    return max(2, target_w), max(2, target_h)


def _cap_output_size(width: int, height: int, max_width: int | None) -> tuple[int, int]:
    if not max_width or width <= max_width:
        return _even_size(width, height)
    scale = max_width / width
    return _even_size(int(round(width * scale)), int(round(height * scale)))


def _even_size(width: int, height: int) -> tuple[int, int]:
    return max(2, width - width % 2), max(2, height - height % 2)


def _fit_to_4_3_smart(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Fit the whole source into 4:3 over a soft contextual background."""

    h, w = frame.shape[:2]
    target = out_w / out_h
    source = w / h

    background = _cover_resize(frame, out_w, out_h)
    background = cv2.GaussianBlur(background, (0, 0), sigmaX=max(12, out_w / 34), sigmaY=max(12, out_h / 34))
    background = cv2.addWeighted(background, 0.68, np.zeros_like(background), 0.32, 0)

    if source > target:
        fit_w = out_w
        fit_h = max(2, int(round(out_w / source)))
    else:
        fit_h = out_h
        fit_w = max(2, int(round(out_h * source)))

    foreground = cv2.resize(frame, (fit_w, fit_h), interpolation=cv2.INTER_AREA)
    x0 = (out_w - fit_w) // 2
    y0 = (out_h - fit_h) // 2

    mask = np.ones((fit_h, fit_w), dtype=np.float32)
    feather = max(8, min(out_w, out_h) // 32)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather, sigmaY=feather)
    if fit_w < out_w or fit_h < out_h:
        inset = max(1, feather // 2)
        mask[:inset] *= np.linspace(0.25, 1.0, inset, dtype=np.float32)[:, None]
        mask[-inset:] *= np.linspace(1.0, 0.25, inset, dtype=np.float32)[:, None]
        mask[:, :inset] *= np.linspace(0.25, 1.0, inset, dtype=np.float32)[None, :]
        mask[:, -inset:] *= np.linspace(1.0, 0.25, inset, dtype=np.float32)[None, :]

    out = background.astype(np.float32)
    region = out[y0 : y0 + fit_h, x0 : x0 + fit_w]
    alpha = mask[:, :, None]
    out[y0 : y0 + fit_h, x0 : x0 + fit_w] = region * (1.0 - alpha) + foreground.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _cover_resize(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max(out_w / w, out_h / h)
    scaled_w = max(out_w, int(round(w * scale)))
    scaled_h = max(out_h, int(round(h * scale)))
    resized = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    x0 = (scaled_w - out_w) // 2
    y0 = (scaled_h - out_h) // 2
    return resized[y0 : y0 + out_h, x0 : x0 + out_w]


def _resolution_degrade(img: np.ndarray, profile, style: str) -> np.ndarray:
    h, w = img.shape[:2]
    factor = 0.58 if style == "VHS 80s" else 0.72
    factor = max(0.24, factor - 0.10 * profile.scale)
    small = cv2.resize(img, (max(2, int(w * factor)), max(2, int(h * factor))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _analog_blur(img: np.ndarray, profile) -> np.ndarray:
    k = max(3, int(profile.blur * 4) * 2 + 1)
    return cv2.GaussianBlur(img, (k, k), sigmaX=profile.blur, sigmaY=profile.blur * 0.65)


def _artistic_black_and_white(img: np.ndarray, frame_index: int, profile, rng) -> np.ndarray:
    """Create a high-contrast noir grade with adaptive subject protection."""

    b, g, r = cv2.split(np.clip(img, 0, 1))
    luma = r * 0.36 + g * 0.46 + b * 0.18
    red_filter = r * 0.52 + g * 0.34 + b * 0.14
    blue_filter = r * 0.18 + g * 0.44 + b * 0.38
    color_contrast = np.clip(np.abs(red_filter - blue_filter) * 1.8, 0, 1)
    y = luma * 0.58 + red_filter * 0.30 + blue_filter * 0.12
    y = _local_contrast_luma(y, clip_limit=3.2)

    h, w = y.shape
    edges = np.abs(cv2.Laplacian(y, cv2.CV_32F, ksize=3))
    edges = cv2.GaussianBlur(np.clip(edges * 5.5, 0, 1), (0, 0), sigmaX=2.2)
    yy, xx = np.ogrid[-1:1:h * 1j, -1:1:w * 1j]
    center_weight = np.exp(-((xx / 0.72) ** 2 + (yy / 0.82) ** 2)).astype(np.float32)
    subject = np.clip(edges * 0.45 + color_contrast * 0.28 + center_weight * 0.34, 0, 1)
    subject = cv2.GaussianBlur(subject, (0, 0), sigmaX=max(5, w / 90), sigmaY=max(5, h / 90))

    local = cv2.GaussianBlur(y, (0, 0), sigmaX=18, sigmaY=18)
    detail = y - local
    shadows = np.clip((0.50 - y) / 0.50, 0, 1)
    highlights = np.clip((y - 0.58) / 0.42, 0, 1)

    y = y + detail * (0.62 + subject * 0.30)
    y = y - shadows * (0.16 + (1.0 - subject) * 0.18)
    y = y + highlights * (0.12 + subject * 0.08)
    y = y + subject * 0.045
    y = _s_curve(y, contrast=1.46, pivot=0.47)

    # Shape the frame like noir lighting: deeper corners and a soft practical-light bloom.
    noir_vignette = 1.0 - np.clip(np.sqrt((xx * 0.92) ** 2 + (yy * 1.04) ** 2) - 0.18, 0, 1) * 0.52
    y *= noir_vignette.astype(np.float32)
    bloom_mask = np.clip((y - 0.72) / 0.28, 0, 1)
    bloom = cv2.GaussianBlur(bloom_mask, (0, 0), sigmaX=7 + profile.scale * 5)
    y = y + bloom * 0.07

    gate = 1.0 + math.sin(frame_index * 0.13) * 0.006
    y = np.clip(y * gate, 0, 1)
    tone = np.dstack([y * 1.015, y * 1.005, y * 0.965])
    shadow_tone = shadows[:, :, None] * np.array([0.018, 0.012, 0.002], dtype=np.float32)
    highlight_tone = highlights[:, :, None] * np.array([0.000, 0.004, 0.016], dtype=np.float32)
    out = tone + shadow_tone + highlight_tone
    return np.clip(out, 0, 1)


def _local_contrast_luma(y: np.ndarray, clip_limit: float) -> np.ndarray:
    y8 = np.clip(y * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(y8).astype(np.float32) / 255.0
    return np.clip(y * 0.48 + enhanced * 0.52, 0, 1)


def _s_curve(y: np.ndarray, contrast: float, pivot: float) -> np.ndarray:
    return np.clip((y - pivot) * contrast + pivot, 0, 1)


def _vhs_signal_pipeline(
    img: np.ndarray,
    frame_index: int,
    fps: float,
    profile,
    state: VHSState,
    rng: np.random.Generator,
) -> np.ndarray:
    """Approximate a VHS signal chain before the image reaches a CRT.

    VHS does not mainly fail as RGB pixels. It stores a relatively detailed
    luma signal and a much weaker color-under chroma signal, so most realistic
    artifacts are easier to model in YCrCb-like luma/chroma space.
    """

    strength = 1.0
    state.previous_clean = img.copy()
    _vhs_update_state(state, profile, rng, strength)

    y, cr, cb = _separate_luma_chroma(img)
    y = _vhs_bandlimit_luma(y, profile, strength)
    cr, cb = _vhs_bandlimit_chroma(cr, cb, profile, strength)
    cr, cb = _vhs_chroma_phase_error(cr, cb, state, profile, strength)
    y = _vhs_luma_ringing(y, profile, strength)
    cr, cb = _vhs_chroma_crosstalk(y, cr, cb, frame_index, state, profile, strength)
    y, cr, cb = _vhs_analog_gain(y, cr, cb, state, profile, strength)
    y, cr, cb = _vhs_temporal_chroma_ghost(y, cr, cb, state.previous_signal, profile, strength)
    y, cr, cb = _vhs_rf_noise(y, cr, cb, profile, rng, strength)
    y, cr, cb = _vhs_dropout_lines(y, cr, cb, profile, rng, state, strength)

    signal = _combine_luma_chroma(y, cr, cb)
    signal = _vhs_luma_smear(signal, profile, strength)
    signal = _vhs_time_base_error(signal, frame_index, profile, state, strength)
    signal = _vhs_head_switching_noise(signal, frame_index, profile, rng, state, strength)
    signal = _vhs_interlaced_fields(signal, state.previous_signal, frame_index, profile, strength)
    state.previous_signal = np.clip(signal, 0.0, 1.0).copy()

    display = _vhs_crt_display(signal, frame_index, profile, strength)
    return np.clip(display, 0.0, 1.0)


def _vhs_update_state(state: VHSState, profile, rng: np.random.Generator, strength: float) -> None:
    """Advance slow tape phases by small amounts.

    Tiny random increments prevent perfect periodicity, but the state makes the
    errors drift across frames instead of resetting as unrelated random damage.
    """

    state.chroma_phase += 0.012 * strength + float(rng.normal(0, 0.0015 * profile.scale))
    state.tracking_phase += 0.018 * strength + float(rng.normal(0, 0.0018 * profile.scale))
    state.head_switch_phase += 0.015 * strength + float(rng.normal(0, 0.0012 * profile.scale))
    state.color_gain_phase += 0.010 * strength

    if state.dropout_burst > 0:
        state.dropout_burst -= 1
    elif rng.random() < profile.dropout * 0.12 * strength:
        state.dropout_burst = int(rng.integers(2, 8 + int(profile.scale * 6)))

    if state.tracking_burst > 0:
        state.tracking_burst -= 1
    elif rng.random() < 0.0015 * profile.scale * strength:
        state.tracking_burst = int(rng.integers(1, 3))


def _separate_luma_chroma(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ycrcb = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_BGR2YCrCb)
    return ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]


def _combine_luma_chroma(y: np.ndarray, cr: np.ndarray, cb: np.ndarray) -> np.ndarray:
    ycrcb = cv2.merge([np.clip(y, 0, 1), np.clip(cr, 0, 1), np.clip(cb, 0, 1)])
    return cv2.cvtColor(ycrcb.astype(np.float32), cv2.COLOR_YCrCb2BGR)


def _vhs_bandlimit_luma(y: np.ndarray, profile, strength: float) -> np.ndarray:
    h, w = y.shape
    # VHS luma is much sharper vertically than horizontally. We model that by
    # lowering horizontal sample count and using anisotropic blur.
    target_w = max(2, min(w, max(16, int(w / (1.75 + 0.55 * profile.scale * strength)))))
    soft = cv2.resize(y, (target_w, h), interpolation=cv2.INTER_AREA)
    soft = cv2.resize(soft, (w, h), interpolation=cv2.INTER_LINEAR)
    sigma_x = 1.20 + 1.25 * profile.scale * strength
    sigma_y = 0.18 + 0.12 * profile.scale * strength
    soft = cv2.GaussianBlur(soft, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
    if h > 360:
        # Only a mild vertical bandwidth touch: enough to remove HD crispness,
        # not enough to make the picture collapse vertically.
        mild_h = max(240, int(h / (1.08 + 0.08 * strength)))
        soft = cv2.resize(soft, (w, mild_h), interpolation=cv2.INTER_AREA)
        soft = cv2.resize(soft, (w, h), interpolation=cv2.INTER_LINEAR)
    return soft


def _vhs_bandlimit_chroma(cr: np.ndarray, cb: np.ndarray, profile, strength: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = cr.shape
    chroma_w = max(2, min(w, max(16, int(w / (3.8 + 1.5 * profile.scale * strength)))))
    chroma_h = max(2, min(h, max(24, int(h / (1.55 + 0.30 * profile.scale * strength)))))

    def process(channel: np.ndarray, delay: int) -> np.ndarray:
        low = cv2.resize(channel, (chroma_w, chroma_h), interpolation=cv2.INTER_AREA)
        low = cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)
        low = cv2.GaussianBlur(low, (0, 0), sigmaX=3.5 + 2.5 * profile.scale * strength, sigmaY=0.75)
        return np.roll(low, delay, axis=1)

    delay = max(1, int(round(profile.chroma_shift * strength)))
    return process(cr, delay), process(cb, -max(1, delay // 2))


def _vhs_chroma_phase_error(cr: np.ndarray, cb: np.ndarray, state: VHSState, profile, strength: float) -> tuple[np.ndarray, np.ndarray]:
    # Color-under phase error rotates chroma around the neutral 0.5 point. This
    # creates hue drift without the fake look of a large RGB split.
    angle = math.sin(state.chroma_phase) * (0.030 + 0.030 * profile.scale) * strength
    angle += math.sin(state.chroma_phase * 0.37 + 1.7) * (0.018 * profile.scale) * strength
    c, s = math.cos(angle), math.sin(angle)
    u = cr - 0.5
    v = cb - 0.5
    return 0.5 + u * c - v * s, 0.5 + u * s + v * c


def _vhs_luma_ringing(y: np.ndarray, profile, strength: float) -> np.ndarray:
    """Add small delayed edge echoes from limited analog luma bandwidth."""

    delay = max(1, int(round(1.0 + profile.scale * 1.7 * strength)))
    edge = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    edge = cv2.GaussianBlur(edge, (0, 0), sigmaX=1.2 + profile.scale, sigmaY=0.18)
    bright = np.clip((y - 0.18) / 0.82, 0, 1)
    echo = np.roll(edge, delay, axis=1) * (0.016 + profile.scale * 0.022) * strength
    pre_echo = np.roll(edge, -delay * 2, axis=1) * (0.005 + profile.scale * 0.007) * strength
    return y + (echo - pre_echo) * (0.55 + bright * 0.45)


def _vhs_chroma_crosstalk(
    y: np.ndarray,
    cr: np.ndarray,
    cb: np.ndarray,
    frame_index: int,
    state: VHSState,
    profile,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Leak high-frequency luma into chroma as soft NTSC/VHS color crawl."""

    h, w = y.shape
    low = cv2.GaussianBlur(y, (0, 0), sigmaX=2.2 + profile.scale * 1.6, sigmaY=0.4)
    high = np.clip(y - low, -0.18, 0.18)
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    carrier = np.sin(xs * 0.58 + ys * math.pi + state.chroma_phase * 5.5 + frame_index * 0.37)
    crawl = high * carrier * (0.055 + profile.scale * 0.035) * strength
    saturation = np.clip(np.abs(cr - 0.5) + np.abs(cb - 0.5), 0.0, 0.42) / 0.42
    crawl *= 0.45 + saturation * 0.55
    return cr + crawl * 0.62, cb - crawl * 0.48


def _vhs_analog_gain(y: np.ndarray, cr: np.ndarray, cb: np.ndarray, state: VHSState, profile, strength: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gain = 1.0 + math.sin(state.color_gain_phase * 1.8) * 0.020 * profile.scale * strength
    gain += math.sin(state.color_gain_phase * 0.57 + 2.0) * 0.012 * profile.scale * strength
    black_lift = 0.030 + 0.020 * profile.scale * strength
    black_lift += math.sin(state.color_gain_phase * 1.25 + 0.8) * 0.006 * profile.scale * strength
    chroma_gain = 0.78 + math.sin(state.color_gain_phase * 1.3 + 1.2) * 0.040 * profile.scale * strength

    y = y * gain
    y = np.where(y > 0.88, 0.88 + (y - 0.88) * (0.55 - 0.08 * profile.scale), y)
    y = y * (0.91 - 0.025 * profile.scale) + black_lift
    cr = 0.5 + (cr - 0.5) * chroma_gain
    cb = 0.5 + (cb - 0.5) * (chroma_gain * 0.92)
    cr += 0.010 * profile.scale * strength
    cb -= 0.006 * profile.scale * strength
    return y, cr, cb


def _vhs_rf_noise(y: np.ndarray, cr: np.ndarray, cb: np.ndarray, profile, rng, strength: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = y.shape
    luma = np.clip(y, 0, 1)
    shadow = np.clip(1.18 - luma, 0.25, 1.35)
    flat = 1.0 - np.clip(np.abs(cv2.Laplacian(luma, cv2.CV_32F, ksize=3)) * 18.0, 0.0, 0.65)

    fine = rng.normal(0, (0.006 + profile.scale * 0.010) * strength, (h, w)).astype(np.float32)
    line = rng.normal(0, (0.006 + profile.scale * 0.006) * strength, (h, 1)).astype(np.float32)
    line = cv2.GaussianBlur(np.repeat(line, w, axis=1), (0, 0), sigmaX=18)
    y = y + (fine + line) * shadow * flat

    chroma_noise = rng.normal(0, (0.010 + profile.scale * 0.010) * strength, (h, w)).astype(np.float32)
    chroma_noise = cv2.GaussianBlur(chroma_noise, (0, 0), sigmaX=5.0, sigmaY=1.4)
    cr = cr + chroma_noise * 0.55
    cb = cb - chroma_noise * 0.45

    impulse_count = int(profile.scale * strength * h * w * 0.000025)
    if impulse_count:
        ys = rng.integers(0, h, impulse_count)
        xs = rng.integers(0, w, impulse_count)
        y[ys, xs] = rng.choice([0.08, 0.95], impulse_count)
    return y, cr, cb


def _vhs_dropout_lines(y: np.ndarray, cr: np.ndarray, cb: np.ndarray, profile, rng, state: VHSState, strength: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = y.shape
    out_y, out_cr, out_cb = y.copy(), cr.copy(), cb.copy()
    burst_boost = 1.8 if state.dropout_burst > 0 else 1.0
    attempts = int((3 + profile.dropout * 55) * strength * burst_boost)
    for _ in range(attempts):
        if rng.random() > 0.22 + profile.dropout * 1.45:
            continue
        row = int(rng.integers(0, h))
        thickness = int(rng.integers(1, max(2, 2 + int(profile.scale * 2))))
        min_len = max(2, min(w, max(8, w // 16)))
        max_len = max(min_len + 1, min(w + 1, int(w * (0.28 + profile.dropout * 1.45))))
        length = int(rng.integers(min_len, max_len))
        x0 = int(rng.integers(0, max(1, w - length)))
        x1 = min(w, x0 + length)
        row_end = min(h, row + thickness)
        segment_w = x1 - x0
        if segment_w <= 0:
            continue

        if segment_w > 2:
            horizontal = 0.18 + 0.82 * np.hanning(segment_w).astype(np.float32)
        else:
            horizontal = np.ones(segment_w, dtype=np.float32)
        feather = np.repeat(horizontal[None, :], row_end - row, axis=0)
        feather = cv2.GaussianBlur(feather, (0, 0), sigmaX=max(1.5, segment_w * 0.018), sigmaY=0.45)
        feather *= rng.uniform(0.28, 0.74)
        noisy_floor = rng.normal(0.0, 0.045 + profile.scale * 0.020, feather.shape).astype(np.float32)
        target = out_y[row:row_end, x0:x1] * rng.uniform(0.48, 0.82) + rng.uniform(0.08, 0.22) + noisy_floor
        if rng.random() < 0.20 + profile.dropout * 0.8:
            target += rng.uniform(0.16, 0.34)
        out_y[row:row_end, x0:x1] = out_y[row:row_end, x0:x1] * (1 - feather) + target * feather
        if rng.random() < 0.55:
            neutral = 0.5 + rng.normal(0.0, 0.025, feather.shape).astype(np.float32)
            out_cr[row:row_end, x0:x1] = out_cr[row:row_end, x0:x1] * (1 - feather * 0.75) + neutral * feather * 0.75
            out_cb[row:row_end, x0:x1] = out_cb[row:row_end, x0:x1] * (1 - feather * 0.75) + neutral * feather * 0.75

    if profile.scale > 1.0 and rng.random() < 0.10 * profile.scale:
        row = int(rng.integers(int(h * 0.12), int(h * 0.88)))
        thickness = int(rng.integers(2, 6 + int(profile.scale * 4)))
        band = slice(row, min(h, row + thickness))
        out_y[band] = cv2.blur(out_y[band], (max(7, int(w * 0.035)), 1))
        out_y[band] += rng.normal(0.10, 0.08, out_y[band].shape)
    return out_y, out_cr, out_cb


def _vhs_temporal_chroma_ghost(
    y: np.ndarray,
    cr: np.ndarray,
    cb: np.ndarray,
    previous_signal: Optional[np.ndarray],
    profile,
    strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if previous_signal is None:
        return y, cr, cb

    prev_y, prev_cr, prev_cb = _separate_luma_chroma(previous_signal)
    prev_y = np.roll(prev_y, 1 + profile.chroma_shift // 2, axis=1)
    prev_cr = np.roll(prev_cr, 3 + profile.chroma_shift, axis=1)
    prev_cb = np.roll(prev_cb, 2 + profile.chroma_shift, axis=1)
    edge = np.abs(cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3))
    bright_edges = np.clip((edge * 2.0 + y - 0.58) / 0.75, 0, 1)
    luma_mix = (0.010 + profile.ghost * 0.10) * bright_edges * strength
    chroma_mix = (0.030 + profile.ghost * 0.33) * strength
    y = y * (1 - luma_mix) + prev_y * luma_mix
    cr = cr * (1 - chroma_mix) + prev_cr * chroma_mix
    cb = cb * (1 - chroma_mix) + prev_cb * chroma_mix
    return y, cr, cb


def _vhs_luma_smear(img: np.ndarray, profile, strength: float) -> np.ndarray:
    """Smear bright transitions horizontally, like overloaded VHS luma."""

    y, cr, cb = _separate_luma_chroma(img)
    highlights = np.clip((y - 0.60) / 0.40, 0, 1)
    smear = np.zeros_like(y)
    taps = (
        (2, 0.030),
        (5, 0.018),
        (9, 0.010),
    )
    for distance, weight in taps:
        amount = weight * (0.75 + profile.scale * 0.55) * strength
        smear += np.roll(y * highlights, distance, axis=1) * amount
    y = y * (1.0 - 0.018 * strength * highlights) + smear
    cr = cr * (1.0 - 0.020 * strength * highlights) + 0.5 * (0.020 * strength * highlights)
    cb = cb * (1.0 - 0.020 * strength * highlights) + 0.5 * (0.020 * strength * highlights)
    return _combine_luma_chroma(y, cr, cb)


def _vhs_time_base_error(img: np.ndarray, frame_index: int, profile, state: VHSState, strength: float) -> np.ndarray:
    h, w = img.shape[:2]
    rows = np.arange(h, dtype=np.float32)
    anchors = max(8, h // 36)
    anchor_y = np.linspace(0, h - 1, anchors)
    anchor_noise = np.sin(anchor_y * 0.031 + state.tracking_phase)
    anchor_noise += 0.55 * np.sin(anchor_y * 0.087 + state.tracking_phase * 0.36 + 1.8)
    anchor_noise *= (0.08 + profile.scale * 0.16) * strength
    smooth = np.interp(rows, anchor_y, anchor_noise)
    drift = math.sin(state.tracking_phase * 0.58) * (0.10 + profile.scale * 0.18) * strength
    slow_bend = np.sin(rows * 0.018 + state.tracking_phase * 0.73) * (0.04 + profile.scale * 0.07) * strength
    offsets = smooth + drift + slow_bend
    return _shift_rows(img, offsets)


def _vhs_head_switching_noise(img: np.ndarray, frame_index: int, profile, rng, state: VHSState, strength: float) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    band_h = max(2, int(h * (0.020 + profile.tracking * 0.22 * strength)))
    drift = int(math.sin(state.head_switch_phase) * h * 0.003)
    y0 = max(0, min(h - band_h, h - band_h - int(h * 0.010) + drift))
    band = out[y0 : y0 + band_h]
    local_rows = band.shape[0]
    rows = np.arange(local_rows, dtype=np.float32)
    offsets = np.sin(rows * 0.65 + state.head_switch_phase * 2.2) * (0.20 + profile.scale * 0.42) * strength
    offsets += np.linspace(0, profile.jitter * 0.22 * strength, local_rows)
    if state.tracking_burst > 0:
        offsets += np.sign(np.sin(state.head_switch_phase)) * (0.55 + profile.jitter * 0.18 * strength)
    band = _shift_rows(band, offsets)
    streaks = rng.normal(0, (0.018 + profile.scale * 0.018) * strength, (local_rows, w)).astype(np.float32)
    streaks = cv2.GaussianBlur(streaks, (0, 0), sigmaX=10 + profile.scale * 16, sigmaY=0.4)
    ramp = np.linspace(0.35, 1.0, local_rows, dtype=np.float32)[:, None, None]
    out[y0 : y0 + band_h] = np.clip(band + streaks[:, :, None] * ramp, 0, 1)
    return out


def _vhs_interlaced_fields(img: np.ndarray, previous_signal: Optional[np.ndarray], frame_index: int, profile, strength: float) -> np.ndarray:
    out = img.copy()
    field = frame_index % 2
    if previous_signal is not None:
        prev = previous_signal
        prev = cv2.resize(prev, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        mix = (0.13 + profile.scale * 0.045) * strength
        out[field::2] = out[field::2] * (1 - mix) + prev[field::2] * mix

    shifted = np.roll(out, max(1, profile.chroma_shift // 2), axis=1)
    out[1 - field :: 2] = out[1 - field :: 2] * 0.82 + shifted[1 - field :: 2] * 0.18
    row_gain = np.ones((out.shape[0], 1, 1), dtype=np.float32)
    row_gain[1::2] -= (0.018 + profile.scale * 0.010) * strength
    row_gain[0::2] += 0.006
    return out * row_gain


def _vhs_crt_display(img: np.ndarray, frame_index: int, profile, strength: float) -> np.ndarray:
    h, w = img.shape[:2]
    # Consumer CRTs normally overscanned the picture, hiding unstable tape
    # edges. This also prevents row-shift edge fills from reading as clean
    # vertical white lines.
    out = _vhs_overscan(img, strength)
    gray = cv2.cvtColor(np.clip(out, 0, 1), cv2.COLOR_BGR2GRAY)
    highlights = np.clip((gray - 0.70) / 0.30, 0, 1)
    bloom = cv2.GaussianBlur(out * highlights[:, :, None], (0, 0), sigmaX=3.0 + profile.scale * 2.8, sigmaY=1.5)
    out = out + bloom * (0.055 + profile.scale * 0.040) * strength

    scan = np.ones(h, dtype=np.float32)
    scan[1::2] -= (0.025 + profile.scale * 0.030) * strength
    out *= scan[:, None, None].astype(np.float32)

    y, x = np.ogrid[-1:1:h * 1j, -1:1:w * 1j]
    radius = np.sqrt((x * 0.94) ** 2 + (y * 1.05) ** 2)
    vignette = 1.0 - np.clip(radius - 0.45, 0, 0.75) * (0.12 + profile.vignette * 0.35) * strength
    out *= vignette[:, :, None]

    corner = np.clip((1.17 - np.maximum(np.abs(x), np.abs(y))) / 0.12, 0, 1)
    return out * corner[:, :, None] + 0.010


def _vhs_overscan(img: np.ndarray, strength: float) -> np.ndarray:
    h, w = img.shape[:2]
    scale = 1.012 + 0.010 * strength
    scaled_w = max(w + 2, int(round(w * scale)))
    scaled_h = max(h + 2, int(round(h * scale)))
    enlarged = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    x0 = (scaled_w - w) // 2
    y0 = (scaled_h - h) // 2
    return enlarged[y0 : y0 + h, x0 : x0 + w]


def _shift_rows(img: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = x - offsets.astype(np.float32)[:, None]
    border_mode = cv2.BORDER_REFLECT_101 if w > 2 else cv2.BORDER_REPLICATE
    return cv2.remap(img, map_x, y, interpolation=cv2.INTER_LINEAR, borderMode=border_mode)


def _film_grain(img: np.ndarray, profile, rng, amount: float = 1.0, emulsion: bool = False) -> np.ndarray:
    luma = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_BGR2GRAY)
    if not emulsion:
        shadow_weight = 1.15 - luma
        noise = rng.normal(0, profile.grain * amount, img.shape).astype(np.float32)
        return img + noise * shadow_weight[:, :, None]

    h, w = img.shape[:2]
    fine = rng.normal(0, 1.0, (h, w)).astype(np.float32)
    medium = rng.normal(0, 1.0, (max(2, h // 2), max(2, w // 2))).astype(np.float32)
    coarse = rng.normal(0, 1.0, (max(2, h // 5), max(2, w // 5))).astype(np.float32)
    medium = cv2.resize(medium, (w, h), interpolation=cv2.INTER_CUBIC)
    coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)

    clumped = (fine * 0.62 + medium * 0.30 + coarse * 0.08)[:, :, None]
    clumped = cv2.GaussianBlur(clumped, (0, 0), sigmaX=0.42)
    if clumped.ndim == 2:
        clumped = clumped[:, :, None]
    shadow_weight = 0.72 + (1.0 - luma) * 0.95
    highlight_weight = 0.42 + np.clip(1.0 - luma * 1.20, 0, 1) * 0.35

    channel_bias = np.array([0.92, 1.00, 1.08], dtype=np.float32)
    chroma = rng.normal(0, profile.grain * amount * 0.16, img.shape).astype(np.float32)
    luminance_grain = clumped * profile.grain * amount * shadow_weight[:, :, None] * channel_bias
    return img + luminance_grain + chroma * highlight_weight[:, :, None]


def _vignette(img: np.ndarray, profile) -> np.ndarray:
    h, w = img.shape[:2]
    y, x = np.ogrid[-1:1:h * 1j, -1:1:w * 1j]
    radius = np.sqrt(x * x + y * y)
    mask = 1.0 - np.clip(radius - 0.35, 0, 0.95) * profile.vignette
    return img * mask[:, :, None]
