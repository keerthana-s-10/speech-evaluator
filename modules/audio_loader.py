"""
audio_loader.py
---------------
Loads any audio file into a normalised RawAcoustics container
that every other module in this project consumes.
"""

import librosa
import numpy as np
from dataclasses import dataclass


@dataclass
class RawAcoustics:
    y:        np.ndarray   # waveform samples (float32, mono)
    sr:       int          # sample rate in Hz
    duration: float        # total duration in seconds


def load_audio(filepath: str, target_sr: int = 22050) -> RawAcoustics:
    """
    Load any audio format supported by librosa (wav, mp3, m4a, ogg, flac).

    Parameters
    ----------
    filepath  : path to the audio file on disk
    target_sr : resample to this rate; 22050 Hz is librosa's default and
                sufficient for all speech-band features we extract.

    Returns
    -------
    RawAcoustics dataclass ready for feature extraction.
    """
    y, sr    = librosa.load(filepath, sr=target_sr, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    return RawAcoustics(y=y, sr=sr, duration=duration)