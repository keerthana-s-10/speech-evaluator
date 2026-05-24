"""
acoustic_analyser.py
--------------------
Extracts four acoustic features from a RawAcoustics object.
Every function returns a float in [0, 1] so values plug directly
into SpeechEvaluator without any further scaling.

Libraries used: librosa, numpy
"""

import numpy as np
import librosa
from .audio_loader import RawAcoustics


# ---------------------------------------------------------------------------
# 1. Pitch Stability
# ---------------------------------------------------------------------------

def extract_pitch_stability(audio: RawAcoustics) -> float:
    """
    Uses librosa's pYIN algorithm to estimate the fundamental frequency (F0)
    on voiced frames, then quantifies stability via coefficient of variation.

    Formula
    -------
        CV        = std(F0_voiced) / mean(F0_voiced)
        stability = clip(1.0 - CV * 2,  0.0, 1.0)

    CV of 0.0 = perfectly monotone  → stability 1.0
    CV >= 0.5 = very erratic        → stability 0.0
    """
    f0, voiced_flag, _ = librosa.pyin(
        audio.y,
        fmin=librosa.note_to_hz("C2"),   # ~65 Hz  — low male voice floor
        fmax=librosa.note_to_hz("C7"),   # ~2093 Hz — high female ceiling
        sr=audio.sr,
    )

    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    if len(voiced_f0) < 10:
        return 0.0  # not enough voiced audio to measure

    cv = np.std(voiced_f0) / (np.mean(voiced_f0) + 1e-9)
    return round(float(np.clip(1.0 - cv * 2, 0.0, 1.0)), 4)


# ---------------------------------------------------------------------------
# 2. Speech Rate
# ---------------------------------------------------------------------------

def extract_speech_rate(word_count: int, duration_seconds: float) -> float:
    """
    Normalises words-per-minute (WPM) to [0, 1].

    Target range  : 130-160 WPM  → score 1.0
    Below  80 WPM : score 0.0    (too slow)
    Above 220 WPM : score 0.0    (too fast)
    Linear ramps between all boundaries.
    """
    if duration_seconds <= 0 or word_count <= 0:
        return 0.0

    wpm            = (word_count / duration_seconds) * 60.0
    IDEAL_LO       = 130.0
    IDEAL_HI       = 160.0
    MIN_WPM        = 80.0
    MAX_WPM        = 220.0

    if IDEAL_LO <= wpm <= IDEAL_HI:
        return 1.0
    elif wpm < IDEAL_LO:
        return round(float(np.clip((wpm - MIN_WPM) / (IDEAL_LO - MIN_WPM), 0.0, 1.0)), 4)
    else:
        return round(float(np.clip(1.0 - (wpm - IDEAL_HI) / (MAX_WPM - IDEAL_HI), 0.0, 1.0)), 4)


# ---------------------------------------------------------------------------
# 3. Energy Consistency
# ---------------------------------------------------------------------------

def extract_energy_consistency(audio: RawAcoustics, frame_length: int = 2048) -> float:
    """
    Measures how controlled the speaker's loudness variation is.

    Process
    -------
    1. Compute per-frame RMS energy.
    2. Smooth with a Hann window to remove micro-level noise.
    3. Score the coefficient of variation (CV) of the smoothed envelope.

    Ideal CV ~0.35 — dynamic enough to be engaging, controlled enough to
    be credible. Both monotone (CV~0) and erratic (CV>>0.5) score lower.
    """
    hop_length  = frame_length // 4
    rms         = librosa.feature.rms(
                      y=audio.y,
                      frame_length=frame_length,
                      hop_length=hop_length
                  )[0]

    win_size    = max(5, len(rms) // 20)
    hann_win    = np.hanning(win_size)
    smoothed    = np.convolve(rms, hann_win / hann_win.sum(), mode="same")

    cv          = np.std(smoothed) / (np.mean(smoothed) + 1e-9)
    IDEAL_CV    = 0.35
    score       = float(np.clip(1.0 - abs(cv - IDEAL_CV) / IDEAL_CV, 0.0, 1.0))
    return round(score, 4)


# ---------------------------------------------------------------------------
# 4. Pause Ratio
# ---------------------------------------------------------------------------

def extract_pause_ratio(audio: RawAcoustics,
                        silence_threshold_db: float = -40.0) -> float:
    """
    Detects silence intervals in the STFT magnitude spectrum, then scores
    the proportion of the recording occupied by meaningful pauses.

    Meaningful pause window : 0.25 s to 2.5 s
      < 0.25 s  micro-gaps / consonant stops, not rhetorical pauses
      > 2.5 s   dead air or technical issues

    Ideal pause ratio : 12% to 22% of total duration.
    """
    hop_length      = 512
    stft            = np.abs(librosa.stft(audio.y, hop_length=hop_length))
    db_per_frame    = np.mean(librosa.amplitude_to_db(stft, ref=np.max), axis=0)
    silence_mask    = db_per_frame < silence_threshold_db
    frame_dur       = hop_length / audio.sr

    pause_durations = []
    in_pause        = False
    count           = 0

    for is_silent in silence_mask:
        if is_silent:
            in_pause = True
            count   += 1
        elif in_pause:
            dur = count * frame_dur
            if 0.25 <= dur <= 2.5:
                pause_durations.append(dur)
            in_pause = False
            count    = 0

    pause_ratio     = sum(pause_durations) / (audio.duration + 1e-9)
    IDEAL_LO        = 0.12
    IDEAL_HI        = 0.22

    if IDEAL_LO <= pause_ratio <= IDEAL_HI:
        return 1.0
    elif pause_ratio < IDEAL_LO:
        return round(float(np.clip(pause_ratio / IDEAL_LO, 0.0, 1.0)), 4)
    else:
        return round(float(np.clip(1.0 - (pause_ratio - IDEAL_HI) / (1.0 - IDEAL_HI), 0.0, 1.0)), 4)